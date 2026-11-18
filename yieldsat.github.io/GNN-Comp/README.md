# YieldSAT GNN-LSTM

The runnable scripts in this directory implement pixel-level yield regression:

```text
Pixel time series [N, T, F]
  -> spatial message-passing GNN + ReLU + dropout (default: 2 layers)
  -> pixel embedding sequences [N, T, 32]
  -> LSTM (default: hidden size 64)
  -> final time-step linear readout
  -> yield predictions [N]
```

## Graph definition

Each node is one pixel. A graph contains pixels from **one field and harvest
year**, identified by `field_shared_name` and `year`. The default graph connects
the eight adjacent positions in the `row`/`col` grid. Use `--connectivity 4` for
horizontal/vertical neighbors only. Edges are bidirectional; missing pixels
are not bridged. Duplicate coordinates inside a field/year raise an error,
since they indicate ambiguous graph grouping.

Each GNN layer applies
`h_next[i] = ReLU(W_self h[i] + W_neighbor mean({h[j]: j -> i, j != i}) + b)`
independently at every time step, followed by dropout. Own features and mean
neighbor messages have separate learned weights. Duplicate edges count once,
self-loops are ignored, and an empty neighborhood contributes zero; isolated
pixels retain their own learned update. The mean-neighbor adjacency is reused
over all time steps and GNN layers. Two layers give a two-hop spatial receptive field;
the LSTM then models each pixel's sequence of spatial embeddings. Targets
are used only in the loss, never to build edges.

Batches concatenate whole graphs with offset node IDs and no edges between
graphs. **Batch size counts field/year graphs, not pixels.** Start with 1.
Memory still scales with the number of pixels in the largest field, time
steps, and hidden dimensions; smaller hidden dimensions can help large fields.
This implementation uses sparse PyTorch operations and does not require
PyTorch Geometric.

## Install and train

From the outer repository root, use a Python environment with these packages:

```powershell
python -m pip install -r yieldsat.github.io/GNN-Comp/requirements.txt
python yieldsat.github.io/GNN-Comp/train.py --folds 5 --epochs 15 --batch-size 1
```

The defaults use `data/` beside `train.py`, select **Germany rapeseed**, and use
only the 12 Sentinel-2 bands as model inputs. Soil, DEM, weather, coordinates,
and yield statistics are not input features. Override the S2 band subset with
`--bands B02 B03 B04 B08`, for example. Feature count follows that selection.
`--data` and `--output-dir` resolve relative to the script directory, independent
of the working directory. Absolute paths are also accepted. The default output
is `yieldsat.github.io/GNN-Comp/runs/germany_rapeseed_s2/`.

The local raw-data layout is:

```text
data/<field-season>/
  metadata-<field-season>.json
  s2_images/S2_L2A_YYYYMMDD.tif
  yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif
```

Metadata selects country/crop before raster loading. Images from seeding through
harvest are averaged into 24 equal-duration intervals (`--time-steps` overrides
this). Nodata observations are excluded from each mean; empty bins become the
existing -1 missing-feature value. S2 values retain their original stored scale;
no additional cloud filtering is applied. Band descriptions identify the bands.
The mean yield raster supplies pixel targets, reprojected with nearest-neighbor
sampling onto the S2 grid, and nodata targets are excluded. S2 grids must match
within each field. Crop/year suffixes are removed from raw field names for CV
grouping, keeping repeated seasons of a field together. All selected sequences
are loaded into memory before training.

A merged NetCDF can still be passed with `--data`; it must include a `country`
variable with string labels or encoded IDs mapped to labels in its attributes.
Temporal bins are used only for raw images. The NetCDF input schema is:

- `sample(index, time_step, band)`: feature sequences; dimension order is handled explicitly.
- `target(index)`: finite pixel yield targets.
- `field_shared_name(index)`: field identity, shared across seasons.
- `year(index)`: harvest year.
- `row(index)`, `col(index)`: integer pixel positions within each field image.
- `crop(index)` with attributes mapping encoded IDs to crop labels.
- A unique `index` coordinate identifying pixels.

Crop and band selection happen before loading feature arrays. Missing or
infinite features are replaced with -1, preserving the original baseline's
input convention. Features otherwise retain their supplied scale; if adding
standardization, fit it on training-fold pixels only. No global dataset
statistics are used to transform held-out data.

Cross-validation groups by `field_shared_name`, keeping all years of a field
in the same fold. Train and validation graphs are built separately after
splitting. The fold count cannot exceed the number of distinct fields.
The default worker count is zero for Windows compatibility. GPU use is
automatic when CUDA is available; `--device cpu` forces CPU execution.

The output directory must be new. Each run writes:

- `config.json`: run settings.
- `fold_0.pt`, etc.: final-epoch weights, model configuration, bands, graph
  connectivity, fill value, crop and split indices.
- `fold_0_predictions.csv`, etc.: final held-out predictions per fold.
- `cv_predictions.csv`: exactly one held-out prediction per pixel.
- `history.csv`: pixel-weighted train/validation MSE and validation R2 by epoch.
- `field_predictions.csv`: mean predictions/targets per field and year.
- `metrics.json`: pixel and field/year RMSE and R2.

Reported predictions use the fixed final epoch, rather than averaging
predictions from all training epochs. Cross-validation metrics are validation
estimates; use a separate test set or nested CV when tuning hyperparameters.

## Evaluate saved predictions

```powershell
python yieldsat.github.io/GNN-Comp/eval.py --predictions yieldsat.github.io/GNN-Comp/runs/germany_rapeseed_s2/cv_predictions.csv --output-dir yieldsat.github.io/GNN-Comp/runs/germany_rapeseed_s2/evaluation --plots
```

This saves metrics, field/year aggregates, scatter plots and yield
histograms. Field aggregation keeps harvest years separate. Duplicate pixel
predictions are rejected to avoid accidentally mixing epochs.

## Use the model directly

From this directory (or with this directory on your Python import path):

```python
import torch
from model import GNN_LSTM_Model
from graphs import build_grid_graph

x = torch.randn(3, 24, 12)  # 3 neighboring pixels, 24 time steps, 12 features
edges = build_grid_graph(rows=[0, 0, 0], cols=[0, 1, 2])
model = GNN_LSTM_Model(input_dim=12)
predictions = model(x, edges)  # [3]; gradients flow through GNN and LSTM
```

Checkpoints from the previous GCN architecture are incompatible with these
message-passing layers. Retrain to produce checkpoints for the updated model.

For checkpoint inference, rebuild graphs with the saved connectivity, band
order, and fill value. Move both samples and edges to the model's device:

```python
checkpoint = torch.load("fold_0.pt", map_location="cpu", weights_only=True)
model = GNN_LSTM_Model(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
with torch.no_grad():
    predictions = model(x, edges)
```

## Verification

```powershell
python -m unittest discover -s yieldsat.github.io/GNN-Comp -v
```

Tests check exact graph aggregation, neighbor influence, disjoint graph
batching, node reordering, gradients through GNN/LSTM/readout, isolated pixels,
field-grouped folds, missing features, dimension ordering, pixel-weighted
losses, evaluation grouping, and a two-fold NetCDF training run with checkpoint
reload. Synthetic tests verify execution and graph semantics, not yield
accuracy on the real dataset.
