"""Field-grouped cross-validation; all seasons of a field stay together."""

from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.model_selection import GroupKFold

# Retain the original import location used by notebook callers.
from data import YieldSATDataset


def split_dataset_by_group(data, n_splits=5, group_key="field_shared_name"):
    if isinstance(data, (str, Path)):
        with xr.open_dataset(data) as source:
            return split_dataset_by_group(source, n_splits, group_key)
    if group_key is None:
        raise ValueError("A grouping key is required to prevent field leakage")
    indices = data.index.values
    groups = data[group_key].transpose("index").values
    if n_splits < 2 or n_splits > len(np.unique(groups)):
        raise ValueError("n_splits must be between 2 and the number of unique groups")
    return [
        (indices[train].tolist(), indices[val].tolist())
        for train, val in GroupKFold(n_splits=n_splits).split(indices, groups=groups)
    ]
