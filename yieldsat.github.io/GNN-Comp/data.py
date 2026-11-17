"""Load YieldSAT without performing file I/O at import time."""

from pathlib import Path

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


def load_dataset(path, crop="rapeseed", bands=DEFAULT_BANDS):
    """Select crop/features before loading, avoiding all 120 bands in memory."""
    with xr.open_dataset(Path(path)) as source:
        selected = filter_ds_by_attribute(source, condition=crop)
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
