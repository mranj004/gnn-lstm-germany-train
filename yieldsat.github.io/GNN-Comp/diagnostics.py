"""Descriptive diagnostics only: these statistics must never fit a model."""

import json
from pathlib import Path
import numpy as np


def distribution(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return dict(count=len(x), min=float(x.min()), max=float(x.max()),
                mean=float(x.mean()), std=float(x.std()),
                percentiles=dict(zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"],
                                     np.percentile(x, [1, 5, 25, 50, 75, 95, 99]).tolist())))


def data_diagnostics(dataset, transform):
    samples = dataset['sample'].transpose('index', 'time_step', 'band').values
    observed_bins = np.zeros(samples.shape[:2], dtype=bool)
    complete_bins = np.ones(samples.shape[:2], dtype=bool)
    bands = {}
    for i, name in enumerate(dataset.band.values):
        x = samples[:, :, i]
        valid = transform.observed(x)
        observed_bins |= valid
        complete_bins &= valid
        bands[str(name)] = dict(distribution(x[valid]), missing_percent=float(100 * (1-valid.mean())),
                               missing_percent_by_time=(100 * (1-valid.mean(axis=0))).tolist())
    return dict(bands=bands, targets=distribution(dataset.target.values),
                any_missing_bin_percent=float(100 * (1-complete_bins.mean())),
                fully_unobserved_bin_percent=float(100 * (1-observed_bins.mean())),
                fully_unobserved_bin_percent_by_time=(100 * (1-observed_bins.mean(axis=0))).tolist(),
                fully_unobserved_pixel_percent=float(100 * (~observed_bins.any(axis=1)).mean()))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
