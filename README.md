# GNN-LSTM for YieldSAT

The implementation and usage instructions are in [yieldsat.github.io/GNN-Comp](yieldsat.github.io/GNN-Comp/README.md).

The model applies spatial GNN message passing to neighboring pixels at each time step, then an LSTM predicts yield from each pixel's sequence. Training uses field-grouped cross-validation and saves checkpoints, held-out predictions, and evaluation metrics.
