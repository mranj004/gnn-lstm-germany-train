"""Load YieldSAT without performing file I/O at import time."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from graphs import build_grid_graph

DEFAULT_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")


def filter_ds_by_attribute(dataset, filter_variable="crop", condition=None):
    if condition is None:
        return dataset
    mapping = {str(value): key for key, value in dataset[filter_variable].attrs.items()}
    if condition not in mapping:
        raise ValueError(f"Unknown {filter_variable} {condition!r}; available: {list(mapping)}")
    value = np.asarray(mapping[condition], dtype=dataset[filter_variable].dtype).item()
    return dataset.sel(index=dataset[filter_variable] == value)


def load_field_images(path, crop="rapeseed", bands=DEFAULT_BANDS, country="Germany", time_steps=24):
    """Average dated S2 observations into equal bins from seeding to harvest."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    if time_steps < 1:
        raise ValueError("time_steps must be positive")
    bands = list(bands or DEFAULT_BANDS)
    if not bands or set(bands) - set(DEFAULT_BANDS):
        raise ValueError("Only Sentinel-2 bands are supported")
    fields = []
    for metadata_path in sorted(Path(path).glob("*/metadata-*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if country is not None and metadata["adm_units"]["country"] != country:
            continue
        if crop is not None and metadata["crop"] != crop:
            continue
        folder = metadata_path.parent
        start = pd.to_datetime(metadata["seeding_date"], format="%d.%m.%Y")
        end = pd.to_datetime(metadata["harvesting_date"], format="%d.%m.%Y")
        if end <= start:
            raise ValueError(f"Invalid growing season in {metadata_path}")
        images = []
        for image in sorted((folder / "s2_images").glob("S2_L2A_*.tif")):
            date = pd.to_datetime(image.stem.removeprefix("S2_L2A_"), format="%Y%m%d")
            if start <= date <= end:
                step = min(int((date - start) / (end - start) * time_steps), time_steps - 1)
                images.append((image, step))
        if not images:
            raise ValueError(f"No growing-season S2 images in {folder}")
        with rasterio.open(images[0][0]) as reference:
            shape, transform, crs = reference.shape, reference.transform, reference.crs
        # Yield rasters have slightly different transforms; align targets to S2 pixels.
        target = np.full(shape, np.nan, dtype=np.float32)
        target_path = folder / "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif"
        with rasterio.open(target_path) as source:
            reproject(
                source=rasterio.band(source, 1), destination=target,
                src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
                dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
        rows, cols = np.where(np.isfinite(target))
        if not len(rows):
            raise ValueError(f"No valid yield pixels in {target_path}")
        sums = np.zeros((len(rows), time_steps, len(bands)), dtype=np.float32)
        counts = np.zeros_like(sums, dtype=np.int32)
        for image, step in images:
            with rasterio.open(image) as source:
                if source.shape != shape or source.transform != transform or source.crs != crs:
                    raise ValueError(f"S2 grids differ within field: {image}")
                labels = [description.split(" - ")[0] if description else None
                          for description in source.descriptions]
                if any(band not in labels for band in bands):
                    raise ValueError(f"Missing S2 band descriptions in {image}: {bands}")
                values = source.read([labels.index(band) + 1 for band in bands], masked=True)
                values = values.astype(np.float32).filled(np.nan)[:, rows, cols].T
            valid = np.isfinite(values)
            sums[:, step] += np.where(valid, values, 0)
            counts[:, step] += valid
        samples = np.full_like(sums, np.nan)
        np.divide(sums, counts, out=samples, where=counts > 0)
        # Raw names include crop/year; remove those suffixes for field-grouped CV.
        field_name = metadata["field_shared_name"]
        suffix = f"_{metadata['crop']}_{metadata['year']}"
        shared_name = field_name.removesuffix(suffix)
        fields.append(xr.Dataset(
            {
                "sample": (("index", "time_step", "band"), samples),
                "target": ("index", target[rows, cols]),
                "field_shared_name": ("index", np.repeat(shared_name, len(rows))),
                "year": ("index", np.repeat(metadata["year"], len(rows))),
                "row": ("index", rows), "col": ("index", cols),
            },
            coords={"index": [f"{field_name}_{r}_{c}" for r, c in zip(rows, cols)],
                    "time_step": np.arange(time_steps), "band": bands},
        ))
        print(f"Loaded {field_name}: {len(rows)} pixels, {len(images)} S2 images", flush=True)
    if not fields:
        raise ValueError(f"No fields match country={country!r}, crop={crop!r} in {path}")
    return xr.concat(fields, dim="index")


def load_dataset(path, crop="rapeseed", bands=DEFAULT_BANDS, country=None, time_steps=24):
    """Select crop/features before loading, avoiding all 120 bands in memory."""
    if Path(path).is_dir():
        return load_field_images(path, crop=crop, bands=bands, country=country, time_steps=time_steps)
    with xr.open_dataset(Path(path)) as source:
        # Load small metadata first: scipy-backed string arrays cannot be fancy-indexed lazily.
        if country is not None and "country" in source:
            source["country"].load()
        selected = filter_ds_by_attribute(source, condition=crop)
        if country is not None:
            if "country" not in selected:
                raise ValueError("NetCDF country filtering requires a country variable")
            if selected.country.dtype.kind in "OUS":
                selected = selected.sel(index=selected.country == country)
            else:
                selected = filter_ds_by_attribute(selected, "country", country)
        if bands:
            selected = selected.sel(band=list(bands))
        variables = ["sample", "target", "field_shared_name", "year", "row", "col"]
        return selected[variables].load()


class YieldSATDataset(Dataset):
    """Each item contains all pixels in one field and harvest year.

    indices are coordinate labels from GroupKFold, not positional offsets.
    No label or feature similarity is used to construct the spatial graph.
    """

    def __init__(self, data, fill_value=-1.0, band=DEFAULT_BANDS,
                 indices=None, connectivity=8):
        super().__init__()
        if isinstance(data, (str, Path)):
            data = load_dataset(data, crop=None, bands=band)
        if indices is not None:
            data = data.sel(index=list(indices))
        if band:
            data = data.sel(band=list(band))
        if data.sizes.get("index", 0) == 0:
            raise ValueError("No pixels selected")
        if not data.index.to_index().is_unique:
            raise ValueError("Pixel index labels must be unique")
        self.ds = data
        self.fill_value = fill_value
        self.num_features = data.sizes["band"]
        self.len_sequence = data.sizes["time_step"]
        self.index_ids = data.index.values.astype(str).tolist()
        self.samples = data["sample"].transpose("index", "time_step", "band").values
        self.targets = data["target"].transpose("index").values.astype(np.float32)
        if not np.isfinite(self.targets).all():
            raise ValueError("Yield targets must be finite")
        metadata = pd.DataFrame({
            key: data[key].transpose("index").values
            for key in ("field_shared_name", "year", "row", "col")
        })
        if metadata.isna().any().any():
            raise ValueError("Graph metadata must not contain missing values")
        self.metadata = metadata
        self.graphs = []
        for positions in metadata.groupby(["field_shared_name", "year"], sort=False).indices.values():
            points = metadata.iloc[positions]
            edges = build_grid_graph(points.row.values, points.col.values, connectivity)
            self.graphs.append((positions, edges))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        positions, edges = self.graphs[idx]
        sample = np.nan_to_num(
            self.samples[positions].astype(np.float32), nan=self.fill_value,
            posinf=self.fill_value, neginf=self.fill_value,
        )
        metadata = self.metadata.iloc[positions]
        return {
            "sample": torch.from_numpy(sample),
            "target": torch.from_numpy(self.targets[positions]),
            "edge_index": edges,
            "index": [self.index_ids[i] for i in positions],
            "field_shared_name": metadata.field_shared_name.tolist(),
            "year": metadata.year.tolist(),
        }
