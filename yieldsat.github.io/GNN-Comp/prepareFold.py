"""Field-grouped cross-validation; all seasons of a field stay together."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Retain the original import location used by notebook callers.
from data import YieldSATDataset, canonicalize_metadata


def split_dataset_by_group(data, n_splits=5, group_key="field_shared_name"):
    if isinstance(data, (str, Path)):
        with xr.open_dataset(data) as source:
            return split_dataset_by_group(source, n_splits, group_key)
    if group_key is None:
        raise ValueError("A grouping key is required to prevent field leakage")
    if group_key == 'field_shared_name':
        data = canonicalize_metadata(data)
    indices = data.index.values
    groups = data[group_key].transpose("index").values
    if not data.index.to_index().is_unique or pd.isna(indices).any() or pd.isna(groups).any():
        raise ValueError("Unique nonmissing indices and nonmissing field groups are required")
    if n_splits < 2 or n_splits > len(np.unique(groups)):
        raise ValueError("n_splits must be between 2 and the number of unique groups")
    return [
        (indices[train].tolist(), indices[val].tolist())
        for train, val in GroupKFold(n_splits=n_splits).split(indices, groups=groups)
    ]


def inner_field_split(data, outer_train_indices, fraction, seed):
    data = canonicalize_metadata(data)
    indices = np.asarray(outer_train_indices)
    groups = data['field_shared_name'].sel(index=indices).values
    if not 0 < fraction < 1 or len(np.unique(groups)) < 2:
        raise ValueError("Inner validation requires 0 < fraction < 1 and at least two training fields")
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(indices, groups=groups))
    return indices[train].tolist(), indices[val].tolist()


def split_manifest(data, splits):
    """Compact field-to-fold assignment, reusable across architectures/seeds."""
    data = canonicalize_metadata(data)
    assignments = {}
    for fold, (_, validation) in enumerate(splits):
        for field in np.unique(data.field_shared_name.sel(index=validation).values):
            key = str(field)
            if key in assignments:
                raise ValueError("Field occurs in multiple validation folds")
            assignments[key] = fold
    return {"version": 1, "field_folds": assignments}


def splits_from_manifest(data, manifest, n_splits):
    data = canonicalize_metadata(data)
    groups = data.field_shared_name.values.astype(str)
    assignments = manifest['field_folds']
    if set(groups) != set(assignments):
        raise ValueError("Split manifest fields do not match selected data")
    if set(assignments.values()) != set(range(n_splits)):
        raise ValueError("Split manifest fold count differs from --folds")
    folds = np.array([assignments[g] for g in groups])
    indices = data.index.values
    return [(indices[folds != f].tolist(), indices[folds == f].tolist()) for f in range(n_splits)]
