# Validation record

Validated on 2026-11-17 with Python in `C:/Users/Owner/anaconda3`, PyTorch
2.8.0+cpu. CUDA was unavailable, so GPU execution is not verified locally.

Incremental gates:

| Stage | Checks | Result |
|---|---|---|
| Original checkout | Existing suite | 14 passed |
| 1 | Diagnostics, normalization/masks, plain LSTM, raw SCL | 17 passed |
| 2 | Residual dimensions and gradients; graph ablations | 18 passed |
| 3 | Directional messages, attention including all-missing bins | 20 passed |
| 4 | Training/selection, all A–H smoke runs, reload, evaluation/bootstrap | 27 passed |

The final suite runs tiny two-fold NetCDF experiments for all eight recipes
and a combined normalized/masked/residual/directional/attention model. It checks
saved predictions against reloaded checkpoints, exported attention, LR history,
and generated plot files. It verifies full-epoch pixel accumulation against a
direct full-pixel gradient and checks partial accumulation windows. Changing
outer targets does not change fold-0 normalization or its inner-selected model.
Encoded NetCDF fields are checked for cross-season separation protection.

The field bootstrap is tested for deterministic results under row reordering,
known paired error differences, and rejection of different targets. A float32
field-aggregation rounding discrepancy found during testing was fixed by
aggregating targets and predictions in float64.

Separate subprocess checks passed for launching one ablation through
`experiments.py`, comparing saved runs with matching provenance, explicitly
flagging external predictions without provenance, and `eval.py --plots`
(pixel/field-year plots and training curves). `git diff --check` passed.

Reproduce the suite from repository root:

```powershell
python -m unittest discover -s yieldsat.github.io/GNN-Comp -v
```

No full real-data cross-validation jobs were run. No official NetCDF or matched
3D-LSTM predictions were downloaded. The only raw-data inspection was read-only;
synthetic smoke outputs were held in temporary directories and removed by the
test harness. These checks establish implementation behavior, not predictive
superiority over any benchmark.
