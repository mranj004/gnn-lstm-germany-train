# YieldSAT GNN-LSTM experiments

PyTorch pixel yield regression for Germany rapeseed using 12 Sentinel-2 bands.
[AUDIT.md](AUDIT.md) separates confirmed findings from hypotheses and protocol
differences. Synthetic tests do not establish real-data accuracy improvements.

## Architectures

| Option | Flow |
|---|---|
| Default `--model gnn_lstm` | S2 → two mean-message GNN layers → LSTM → final state → linear yield |
| `--model lstm` | S2 → LSTM → final state → linear yield |
| `--residual` | Concatenate original inputs and GNN embeddings before the LSTM |
| `--temporal-pool attention` | Learned softmax pooling of LSTM outputs before readout |
| `--aggregation directional` | Direction-specific neighbor transforms |

Defaults: 32 GNN channels, 64 LSTM hidden channels, two GNN layers, one LSTM
layer, dropout 0.1. Configure with `--gnn-hidden-dim`, `--hidden-dim`,
`--gnn-layers`, `--lstm-layers`, and `--dropout`. Residual concatenation adjusts
LSTM input size automatically and includes observation masks if appended.

Graphs contain existing labelled pixels within one field and harvest year.
Eight neighbors are default; `--connectivity 4` removes diagonals. Missing pixels
are not bridged; duplicate coordinates are rejected. Batches form disjoint
graphs. Targets never set edge weights; node coverage follows valid target
pixels, as in the original pipeline. Unlabelled context nodes are not added.

Default layers compute
`ReLU(W_self x_i + W_neighbor mean(x_j : j -> i, j != i) + b)` plus dropout.
Distinct neighbors count once; self loops are ignored; isolated nodes keep
their self update. Directional layers sum separate transforms for each relative
`(row_src-row_dst, col_src-col_dst)`, divided by total distinct degree. Image
directions need not be geographic bearings on rotated grids. This is
position-aware message passing, not graph attention.

Temporal attention uses `softmax_t(w^T h_t)`; a shared scalar bias is omitted
because it cancels. `--attention-mask` excludes bins with no observed band at
that pixel; partially observed bins remain eligible. All-missing sequences get
zero weights/context and the readout bias. Missing steps still pass through the
LSTM recurrence: masking affects pooling only and follows the pixel's own
observations, even if neighbors are observed. `--export-attention` writes
`attention_0`, etc. in held-out prediction CSVs.

## Installation and input representations

From the repository root:

```powershell
python -m pip install -r yieldsat.github.io/GNN-Comp/requirements.txt
cd yieldsat.github.io/GNN-Comp
```

Training `--data`/`--output-dir` resolve relative to `train.py`; other paths,
including `--splits`, resolve relative to the working directory. Absolute paths
are accepted. Country/crop defaults are Germany/rapeseed.

Raw layout:

```text
data/<field-season>/
  metadata-<field-season>.json
  s2_images/S2_L2A_YYYYMMDD.tif
  scl_masks/S2_L2A_SCL_YYYYMMDD.tif
  yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif
```

Metadata filters fields before raster loading. Band descriptions select the
ordered inputs `B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12`; `--bands`
selects a subset. Only requested S2 bands are read. No soil, weather, DEM,
coordinates or yield statistics enter the input tensor.

The preserved raw baseline averages observations in 24 equal-duration
seeding-to-harvest bins (`--time-steps` overrides this). Nodata/nonfinite
observations are excluded; empty bins remain missing. S2 scales remain as
stored unless normalization is enabled. The existing mean dry-yield raster is
nearest-neighbor reprojected onto the S2 grid; nonfinite targets are excluded.
S2 grids must agree within a season. Targets are never feature-normalized.
Known crop/year suffixes are stripped from field names to group repeated seasons.

### SCL masking

`--cloud-mask` nearest-neighbor reprojects paired SCL files and masks before
temporal averaging. Default exclusions:

| Class | Meaning |
|---|---|
| 0 | No data; always excluded when masking is enabled |
| 1 | Saturated/defective |
| 3 | Cloud shadows |
| 8 | Medium-probability cloud |
| 9 | High-probability cloud |
| 10 | Thin cirrus |
| 11 | Snow/ice |

Class definitions follow [Copernicus SCL documentation](https://documentation.dataspace.copernicus.eu/FAQ.html).

Classes 2, 4, 5, 6 and 7 remain eligible by default. Class 7 is unclassified;
test `--scl-exclude 0 1 3 7 8 9 10 11` as a stricter variant. Missing SCL files
raise an error. Masking preserves target coverage. Unmasked raw processing
remains the default baseline.

### Supplied preprocessed NetCDF

```powershell
python train.py --data C:/YieldSAT/preprocessed-24-ts/Germany/merge_s2-soil-dem-weather-coords.nc --output-dir runs/official_A
```

Required schema: `sample(index,time_step,band)`, `target(index)`,
`field_shared_name(index)`, `year(index)`, `row(index)`, `col(index)`,
`crop(index)` and `country(index)`; pixel indices must be unique. Dimension order
is explicit. String categories or encoded ID-to-label attributes are supported.
Field/year codes are decoded and known crop/year suffixes removed. Only selected
bands/crop/country load. Supplied time steps and targets are preserved;
`--time-steps` applies only to raw images. SCL masking NetCDF inputs is rejected.

**The representations differ.** The
[official tutorial](https://yieldsat.github.io/notebooks/yieldsat-ml-tutorial.html)
describes two calendar years, harvest in year two, least-cloudy image selection,
and masking outside the growing season. Raw growing-season means are not
equivalent. Use a supplied official NetCDF to test that representation; verify
target/grid/field identity first. The [project](https://yieldsat.github.io/)
provides a dataset request link. Nothing downloads automatically.

### Normalization, missingness and diagnostics

Default behavior replaces nonfinite features with -1 and preserves valid scale.
`--normalize` fits per-band population mean/std across training pixels/time only.
With inner validation, only inner-training data fit the transform. Missing
values never enter statistics and become **zero after normalization**.
Constant/unobserved bands use scale 1, with unobserved mean 0; counts are saved.
`--append-mask` appends a binary observed flag per band/time, independently of
normalization. `--missing-value -1` explicitly marks a finite input sentinel;
none is inferred automatically. Raw nodata and NaN/inf are always missing.

Transforms are checkpointed and applied per graph. Fold datasets share source
samples. Normalization uses bounded chunks; diagnostics process one band at a
time. All selected samples still occupy RAM; this is not an out-of-core loader.

```powershell
python train.py --diagnostics-only --output-dir runs/raw_diagnostics
python train.py --diagnostics-only --cloud-mask --output-dir runs/scl_diagnostics
```

Reports include band min/max/mean/std/percentiles, missing percentages by
band/time, any-missing and fully-unobserved bins, and yield distributions.
Training additionally records per-fold training/inner/outer target distributions,
field sizes, prediction distributions, variance ratios, bias and correlation.
Whole-data diagnostics are descriptive and never fit transforms.

## Reproduction commands

Output directories must be new. Defaults preserve original architecture,
preprocessing, graph-batch optimization, Adam lr 0.001, clip norm 5 and fixed
15-epoch selection. Seeds now reset to `seed + fold`, isolating later folds
from earlier stopping. This reproduces the baseline protocol, not bit-for-bit
historical checkpoints from the previous global RNG sequence.

```powershell
# Original GNN-LSTM; saves reusable splits
python train.py --folds 10 --epochs 15 --batch-size 1 --output-dir runs/A
# Plain LSTM, identical representation and splits
python train.py --model lstm --folds 10 --epochs 15 --splits runs/A/splits.json --output-dir runs/B
# Residual GNN-LSTM, identical representation and splits
python train.py --residual --folds 10 --epochs 15 --splits runs/A/splits.json --output-dir runs/F
# Separate normalized/masked comparison family
python train.py --model lstm --normalize --append-mask --splits runs/A/splits.json --output-dir runs/B_normalized_mask
python train.py --residual --normalize --append-mask --splits runs/A/splits.json --output-dir runs/F_normalized_mask
```

CUDA is automatic; `--device cpu` forces CPU. Batch size counts whole field/year
graphs; start at 1. Accumulation avoids combining many graphs but cannot shrink
a single graph; reduce hidden dimensions for large fields. Windows workers
default to zero; multiple workers may duplicate RAM. Seeds, versions, code
hashes and input fingerprints are saved. CUDA hardware/library changes may
prevent bitwise reproducibility.

## Optimization and selection

* `--objective graph_batch`: original batch-mean loss, averaging gradients over
  each accumulation window. With batch size/accumulation 1, fields get equal
  update opportunities regardless of size.
* `--objective pixel`: accumulate squared-error sums and divide by
  `window_batches * (training_pixels / batches_per_epoch)`. This fixed expected
  size preserves pixel weights even with one graph per update. Uniformly
  shuffled batches estimate the pixel objective. Adam/clipping still affect
  the sequence of updates.
* `--accumulation-steps K`: update every K batches; partial final windows use
  their actual batch count. `0` accumulates a whole epoch, matching full
  pixel-MSE gradients before clipping in pixel mode, with one update per epoch.
  Previous computation graphs are released after each backward pass.
* `--gradient-clip 5`: clip after gradient normalization; 0 disables clipping.
  Pre-clip norms/update counts are logged. `--weight-decay` defaults to 0.
* `--scheduler step --scheduler-patience 5 --lr-factor 0.5` uses fixed intervals.
  `--scheduler plateau` requires inner validation and uses its MSE.

Training MSE is pixel-weighted over predictions encountered during training
(with dropout and changing weights), not a post-epoch evaluation loss.

### Training CSV exports

Every run writes `history.csv` after each completed epoch. Columns include
`cv_fold`, `epoch`, `train_mse`, `train_rmse`, `train_r2`, `train_pixels`,
`inner_val_mse`, `inner_val_r2`, `learning_rate`, `gradient_norm_mean`,
`gradient_norm_max`, and `optimizer_steps`. Training RMSE/R2 use the same
predictions encountered during the epoch as training MSE. R2 is blank when
targets are constant; inner-validation columns are blank when disabled.

For actual versus predicted training yields, enable the optional export:

```powershell
python train.py --residual --export-training-predictions --output-dir runs/residual_csv
```

Each completed fold also writes `fold_0_training_predictions.csv`, etc., with
`index`, `field_shared_name`, `year`, `target`, `prediction`, `cv_fold`, `epoch`,
`split` (always `train`), and `residual` (prediction minus target). This costs
one additional inference pass per fold, with dropout disabled, using the saved
selected checkpoint. Only fitting pixels are included: inner/outer validation
pixels are excluded. Training predictions measure fit, not held-out performance.
They are saved per fold because a pixel can be trained on in several folds;
do not merge them into the one-prediction-per-pixel OOF evaluation file.
`--export-attention` also includes attention weights when configured.

```powershell
python train.py --normalize --residual --objective pixel --accumulation-steps 4 --inner-val-fraction 0.2 --early-stopping-patience 8 --epochs 100 --scheduler plateau --scheduler-patience 3 --splits runs/A/splits.json --output-dir runs/inner_selected
```

Fixed-budget runs fit all outer-training fields and select the final epoch.
`--inner-val-fraction > 0` reserves whole fields using GroupShuffleSplit, fits
normalization only on inner-training fields, and selects best inner MSE.
`--min-delta` defines improvement. Patience 0 disables stopping but still selects
best inner MSE if an inner split exists. At least two outer-training fields are
needed. Outer folds are evaluated once after selection; they never drive
stopping/scheduling. The selected model is **not refitted** on all outer-training
fields. Compare the same selection protocol and update budget within each pair.

## Controlled ablations

[ablations.json](ablations.json) configures A–H. No expensive suite runs automatically.

| ID | Change | Direct control |
|---|---|---|
| A | Original two-layer eight-neighbor mean GNN-LSTM | — |
| B | Plain LSTM | A |
| C | Fold-local normalization | A |
| D | One GNN layer | A |
| E | Four neighbors | A |
| F | Residual concatenation | A |
| G | Residual plus temporal attention | F |
| H | Direction-specific messages | A |

```powershell
python experiments.py --experiment A --dry-run
python experiments.py --experiment C --splits runs/A/splits.json
python experiments.py --experiment D --splits runs/A/splits.json
python experiments.py --experiment G --splits runs/A/splits.json
# Masking attention is an additional, separately labelled change
python experiments.py --experiment G --splits runs/A/splits.json --attention-mask --export-attention --output-dir runs/G_masked
```

Select B/E/F/H similarly. Extra arguments intentionally override recipes and
are recorded. Use a new output directory per seed/representation. Compare A/B
with normalization, then A/B with normalization plus SCL, to isolate causes.
Keep targets, splits, bands, optimizer, epoch/selection budget and seed fixed.

## Outputs, evaluation and paired comparison

```text
runs/<experiment>/
  config.json, environment.json, splits.json
  data_diagnostics.json, fold_0_diagnostics.json, ...
  fold_0.pt, ...                  # selected model/transform/indices/config
  fold_0_predictions.csv, ...
  cv_predictions.csv, history.csv
  metrics.json                    # pooled OOF
  per_fold_metrics.csv, fold_summary.json
  train_mean_baseline_metrics.json
  field_predictions.csv
  pixel_evaluation.png, field_year_evaluation.png, training_curves.png  # --plots
```

Both levels report MSE, RMSE, R2, Pearson, bias, prediction/target variance ratio
and distributions. Undefined constant-target R2 or constant-series correlation
is null. Fold summaries report unweighted fold means/stds and defined-fold
counts, distinct from pooled scores. Checkpoints include selected epoch,
parameter count, architecture, ordered bands, preprocessing and train/inner/
outer indices. These are inference checkpoints, not resumable optimizer states.

```powershell
python eval.py --predictions runs/A/cv_predictions.csv --output-dir runs/A/evaluation --plots
python compare.py --a runs/A --b runs/B --output comparisons/A_vs_B.json --bootstrap 2000 --seed 0
```

Plots contain scatter, yield distributions, residual histograms and training
curves (adjacent `history.csv` or `--history`). Comparison requires matching
pixel IDs, targets, field/year metadata and fold assignments. Input provenance
differences are explicitly labelled. Percentile intervals for B−A resample
whole fields, all seasons and pixels together. At least two fields are needed;
fewer than ten is flagged. Intervals condition on fitted OOF predictions, omit
refitting/seed uncertainty and CV training-set dependence, and do not correct
for multiple experiments. Repeat seeds; do not independently bootstrap pixels.

External 3D-LSTM CSVs need `index,field_shared_name,year,target,prediction,cv_fold`.
Absent provenance, `--allow-unverified-protocol` explicitly records the
limitation but does not bypass target/split checks. Published summary metrics
cannot serve as paired predictions. The [paper](https://arxiv.org/html/2604.00940v1)
describes region-stratified field grouping and fold-averaged metrics, unlike
the default unstratified GroupKFold here.

### Checkpoint inference

```python
from train import load_checkpoint, model_validation
from data import load_dataset, YieldSATDataset
from graphs import collate_graphs
from torch.utils.data import DataLoader
import torch

model, transform, ckpt = load_checkpoint('runs/F/fold_0.pt', device='cpu')
source = load_dataset('data', bands=ckpt['bands'], country=ckpt['country'],
    crop=ckpt['crop'], time_steps=ckpt['time_steps'], cloud_mask=ckpt.get('cloud_mask', False),
    scl_exclude=ckpt.get('scl_exclude', [0, 1, 3, 8, 9, 10, 11]))
held_out = YieldSATDataset(source, band=None, indices=ckpt['val_indices'],
    transform=transform, connectivity=ckpt['connectivity'])
predictions, _ = model_validation(model, DataLoader(held_out, batch_size=1,
    collate_fn=collate_graphs), torch.nn.MSELoss(), 'cpu')
```

Use the same representation as training. Never refit transforms on inference
data. Existing mean-message GNN checkpoints remain loadable with defaults;
older GCN checkpoints remain incompatible.

## Interpretation and tests

If A and B both fail, prioritize representation, scaling, observations,
optimization and target shifts. If B beats A on identical inputs, F/D/E test
information loss or excessive mixing. If C helps A, repeat normalization for
B/F before crediting spatial changes. Compare F/G and A/H independently.
Inspect prediction variance and the training-mean baseline: negative R2 alone
does not demonstrate collapse. Official NetCDF comparisons are representation
experiments until target/input equivalence is verified. Repeated tuning against
outer scores requires an untouched test set or nested model selection for a
final unbiased performance claim.

From the repository root:

```powershell
python -m unittest discover -s yieldsat.github.io/GNN-Comp -v
```

Tests cover graph semantics, gradients, residual dimensions, fold-only
normalization, masks, SCL, encoded metadata, splits, pixel-gradient reference
checks, tiny runs of all eight ablations, checkpoint reload, attention exports,
inner-only selection, plots and reproducible field bootstrap. They verify
implementation and experimental protocol, not real-data accuracy.
