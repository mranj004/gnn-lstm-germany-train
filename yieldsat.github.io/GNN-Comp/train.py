"""Train GNN-LSTM with field-grouped cross-validation."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from data import DEFAULT_BANDS, YieldSATDataset, load_dataset
from graphs import collate_graphs
from model import GNN_LSTM_Model
from prepareFold import split_dataset_by_group

SCRIPT_DIR = Path(__file__).resolve().parent


def model_training(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, count = 0.0, 0
    for batch in train_loader:
        sample = batch["sample"].to(device)
        target = batch["target"].to(device)
        edges = batch["edge_index"].to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(sample, edges)
        loss = criterion(predictions, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite; check inputs and learning rate")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * target.numel()
        count += target.numel()
    if not count:
        raise ValueError("Training loader is empty")
    return total_loss / count


@torch.no_grad()
def model_validation(model, val_loader, criterion, device, BEST_R2_SCORE=-float("inf")):
    model.eval()
    total_loss, count = 0.0, 0
    frames = []
    for batch in val_loader:
        target = batch["target"].to(device)
        predictions = model(batch["sample"].to(device), batch["edge_index"].to(device))
        loss = criterion(predictions, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Validation loss is not finite")
        total_loss += loss.item() * target.numel()
        count += target.numel()
        frames.append(pd.DataFrame({
            "index": batch["index"],
            "field_shared_name": batch["field_shared_name"],
            "year": batch["year"],
            "target": target.cpu().numpy(),
            "prediction": predictions.cpu().numpy(),
        }))
    if not count:
        raise ValueError("Validation loader is empty")
    frame = pd.concat(frames, ignore_index=True)
    score = float(r2_score(frame.target, frame.prediction)) if count > 1 else float("nan")
    frame.attrs.update(mse=total_loss / count, r2=score)
    return frame, max(BEST_R2_SCORE, score) if np.isfinite(score) else BEST_R2_SCORE


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="Field data directory or merged NetCDF; relative to this script")
    parser.add_argument("--country", default="Germany")
    parser.add_argument("--crop", default="rapeseed", help="Crop label in crop.attrs")
    parser.add_argument("--bands", nargs="+", choices=DEFAULT_BANDS, default=list(DEFAULT_BANDS))
    parser.add_argument("--time-steps", type=int, default=24, help="Growing-season bins for raw S2 images")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of whole field/year graphs")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/germany_rapeseed_s2"),
                        help="New output directory; relative to this script")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive; num-workers must be nonnegative")
    if args.time_steps < 1:
        parser.error("time-steps must be positive")
    for name in ("data", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, SCRIPT_DIR / value)
    return args


def run_cross_validation(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    dataset = load_dataset(args.data, crop=args.crop, bands=args.bands,
                           country=args.country, time_steps=args.time_steps)
    splits = split_dataset_by_group(dataset, args.folds)
    # A run has its own directory so checkpoints and metrics cannot be overwritten accidentally.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    predictions, history = [], []
    for fold, (train_indices, val_indices) in enumerate(splits):
        train_dataset = YieldSATDataset(dataset, indices=train_indices, band=None, connectivity=args.connectivity)
        val_dataset = YieldSATDataset(dataset, indices=val_indices, band=None, connectivity=args.connectivity)
        loader_options = dict(
            batch_size=args.batch_size, collate_fn=collate_graphs,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        train_loader = DataLoader(
            train_dataset, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + fold), **loader_options,
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
        model_config = dict(
            input_dim=train_dataset.num_features, hidden_dim=args.hidden_dim,
            num_layers=args.lstm_layers, gnn_hidden_dim=args.gnn_hidden_dim,
            gnn_layers=args.gnn_layers, dropout=args.dropout,
        )
        model = GNN_LSTM_Model(**model_config).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_r2 = -float("inf")
        for epoch in range(1, args.epochs + 1):
            train_mse = model_training(model, train_loader, criterion, optimizer, device)
            val_frame, best_r2 = model_validation(model, val_loader, criterion, device, best_r2)
            history.append({
                "cv_fold": fold, "epoch": epoch, "train_mse": train_mse,
                "val_mse": val_frame.attrs["mse"], "val_r2": val_frame.attrs["r2"],
            })
            print(
                f"Fold {fold + 1}/{args.folds}, epoch {epoch}/{args.epochs}: "
                f"train MSE={train_mse:.4f}, val MSE={val_frame.attrs['mse']:.4f}, "
                f"val R2={val_frame.attrs['r2']:.4f}", flush=True,
            )
        # One final-epoch out-of-fold prediction per pixel, not an average over epochs.
        val_frame["cv_fold"] = fold
        val_frame["epoch"] = args.epochs
        predictions.append(val_frame)
        val_frame.to_csv(args.output_dir / f"fold_{fold}_predictions.csv", index=False)
        torch.save({
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": model_config, "bands": list(args.bands),
            "connectivity": args.connectivity, "crop": args.crop,
            "country": args.country, "time_steps": args.time_steps,
            "temporal_aggregation": "growing_season_mean" if args.data.is_dir() else "preprocessed",
            "fill_value": train_dataset.fill_value, "cv_fold": fold, "epoch": args.epochs,
            "train_indices": train_indices, "val_indices": val_indices,
        }, args.output_dir / f"fold_{fold}.pt")
    results = pd.concat(predictions, ignore_index=True)
    results.to_csv(args.output_dir / "cv_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    from eval import evaluate_predictions
    metrics, fields = evaluate_predictions(results)
    fields.to_csv(args.output_dir / "field_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return results


if __name__ == "__main__":
    run_cross_validation(parse_args())
