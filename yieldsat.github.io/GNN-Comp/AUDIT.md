# YieldSAT implementation audit (2026-11-17)

## Evidence and interpretation

The user's 10-fold results (302,847 pixels, 111 field-years; pixel RMSE
1.6952/R2 -0.0286, field-year RMSE 1.3721/R2 -0.0449) are supplied results,
not rerun here. Negative R2 alone does not prove collapsed predictions;
prediction variance, bias and correlation must be inspected. No saved real-run
predictions/checkpoints were supplied in this checkout's implementation folder.

### Confirmed implementation/protocol differences

* **Temporal representation is different.** The checked-in official tutorial
  `../notebooks/yieldsat-ml-tutorial.ipynb` describes 24 steps over two calendar
  years with harvest in the second year, least-cloudy observation selection,
  and masking outside the growing season. `load_field_images` instead averages
  all observations in 24 equal-duration seeding-to-harvest bins. Neither SCL
  masking nor normalization makes those representations equivalent. We preserve
  the raw representation and accept a supplied official NetCDF without resampling.
* **Clouds were included in averages.** Paired `scl_masks/S2_L2A_SCL_<date>.tif`
  files are present locally. The original loader ignored them. This is an omitted
  quality-control option, not a demonstrated explanation of the entire score gap.
* **Stored S2 scale reaches thousands.** A read-only sample of
  `Germany_DUP3_farm1_field11_rapeseed_2016/S2_L2A_20150823.tif` has 12 correctly
  described bands, nodata=0, scale=1, offset=0; valid values range 120–6668
  (median across bands/pixels 2872). Its SCL contains class 9 clouds. This sample
  is not a whole-dataset diagnostic. Unscaled inputs may saturate LSTM gates;
  fold-local normalization is a testable remedy, not a proven accuracy gain.
* **Optimization/reporting weights differ.** Original updates minimize each
  graph-batch mean MSE with one Adam step per batch. With batch size one, small
  and large fields get equal update opportunities. Reported epoch MSE weights
  every pixel equally. The logged loss calculation is correct, but describes a
  different weighting from equal graph updates. Clipping/Adam further prevent
  interpreting the sequence of updates as exact global full-batch minimization.
* **Published protocol is not identical.** The paper states region-stratified,
  field-grouped CV and fold-average metrics. This code uses unstratified
  GroupKFold and pooled OOF metrics. Both are legitimate, different estimands;
  do not attribute their score difference solely to architecture. New evaluation
  exports pooled, per-fold, and unweighted fold-summary metrics separately.
* **Memory duplication.** Original fold construction selected whole feature
  arrays for training/validation, copying the already loaded dataset. Fold
  datasets now share the original sample tensor and index individual graphs.
* **Coverage gaps.** Original evaluation lacked MSE, correlation, distributions,
  per-fold metrics, matched comparison, and uncertainty estimates. Original
  checkpoints had no preprocessing state because no fitted transforms existed.
* **Encoded NetCDF field identities needed canonicalization.** The overview
  notebook's saved output shows `field_shared_name` codes map to names including
  crop/year suffixes, and year codes map to calendar years. The old NetCDF path
  retained codes without decoding/stripping those suffixes, unlike the raw path.
  This could split repeated seasons of the same named field across folds. The
  loader now decodes complete mappings and removes known crop/year suffixes.
  This fixes a potential leakage path, not an explanation for poor raw-data
  scores. Physical field identity still depends on the release's naming quality.

### Checks that passed in the original implementation

Band descriptions determine order; only requested S2 bands are read. Finite
targets are nearest-neighbor reprojected to the S2 grid, and nodata targets are
excluded. Nodata observations are excluded from bin means; empty bins become
NaN then -1. Graph edges connect adjacent labelled pixels within field/year,
with separate self and mean-neighbor transforms, deduplicated edges, no
self-neighbor double count, and no edges between graphs in a batch. Features
are propagated at each time step, then the LSTM processes each pixel sequence.
Tests verify neighbor influence, equivariance and gradients. There was no
evidence of a disconnected GNN or axis swap.

Raw field names have crop/year suffixes removed for CV. GroupKFold keeps all
seasons of that identity together; graphs are built within each partition.
There is no target-derived edge weight. Graph node availability *does* follow
valid target coverage, as in the original labelled-pixel task: this is not a
full-field inference implementation with unlabelled context nodes. Field-year
evaluation keeps years separate and rejects duplicate pixel predictions.
Final-epoch predictions/checkpoints were saved, with no outer-fold checkpoint
selection, despite an unused running `best_r2` variable. Defaults remain fixed
budget; any adaptive selection now uses a separate inner field split.

### Architectural hypotheses, not diagnosed bugs

Mean mixing may discard useful local spectral contrasts; ReLU/dropout may remove
information; two spatial hops may oversmooth; equal neighbor weights ignore
direction; a final-step readout may underuse earlier growth stages. Plain LSTM,
residual concatenation, one layer, four neighbors, direction-specific weights
and attention test these explanations separately. More parameters do not
guarantee better yield prediction.

### Benchmark limitations

No official NetCDF or matched 3D-LSTM predictions were identified in the local
implementation directory. Dataset access is offered through the official
project's request link; no large dataset was downloaded. Exact published
splits, training budgets, checkpoint selection, spatial context, target grid,
and preprocessing must be matched before claiming reproduction. The project
page describes 13 bands, whereas the local images/tutorial use 12 without B10;
this implementation explicitly records the selected 12-band order.

Sources: [official tutorial](https://yieldsat.github.io/notebooks/yieldsat-ml-tutorial.html),
[project and data access](https://yieldsat.github.io/),
[paper, sections 4.5 and 5](https://arxiv.org/html/2604.00940v1).

## Controlled interpretation

First compare A/B on identical inputs and splits; then A/C and B with/without
normalization. Test SCL separately for both models. Compare raw and supplied
official data as *representation* experiments, verifying target/grid identity.
Compare A/F and F/G for residual/attention effects, A/D and A/E for graph extent,
A/H for direction. Repeat selected comparisons across seeds with fixed split
manifests. Compare training-fold mean predictions to all models. Use field
clusters (all seasons together) for paired uncertainty, not independent pixels.
Outer-fold results guide research but repeated tuning against them forfeits an
unbiased final performance claim; reserve a final test set or nested selection.
