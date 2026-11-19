"""Train GNN-LSTM with field-grouped cross-validation."""

import argparse
import json
from pathlib import Path
import random
import hashlib
import platform
import subprocess
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from data import DEFAULT_BANDS, DEFAULT_SCL_EXCLUDE, YieldSATDataset, load_dataset
from preprocessing import SpectralTransform
from diagnostics import data_diagnostics, distribution, write_json
from graphs import collate_graphs
from model import GNN_LSTM_Model
from prepareFold import split_dataset_by_group, inner_field_split, split_manifest, splits_from_manifest

SCRIPT_DIR = Path(__file__).resolve().parent


def forward_batch(model, batch, device, return_attention=False):
    positions = batch.get('positions')
    mask = batch.get('time_mask')
    return model(batch['sample'].to(device), batch['edge_index'].to(device),
                 positions=None if positions is None else positions.to(device),
                 time_mask=None if mask is None else mask.to(device), return_attention=return_attention,
                 graph_ptr=batch.get('graph_ptr'))


def model_training(model, train_loader, criterion, optimizer, device,
                   objective="graph_batch", accumulation_steps=1, gradient_clip=5.0, stats=None):
    model.train()
    total_loss, count = 0.0, 0
    target_sum, target_square_sum = 0.0, 0.0
    if objective not in ("graph_batch", "pixel") or accumulation_steps < 0:
        raise ValueError("Invalid objective or accumulation_steps")
    if not len(train_loader):
        raise ValueError("Training loader is empty")
    accumulation_steps = accumulation_steps or len(train_loader)
    mean_batch_pixels = len(train_loader.dataset.targets) / len(train_loader)
    optimizer.zero_grad(set_to_none=True)
    norms, window_batches = [], 0
    for batch_number, batch in enumerate(train_loader, 1):
        target = batch["target"].to(device)
        predictions = forward_batch(model, batch, device)
        loss = criterion(predictions, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite; check inputs and learning rate")
        # Fixed expected-pixel denominator preserves size weights even at batch_size=1.
        (loss * (target.numel() if objective == "pixel" else 1)).backward()
        window_batches += 1
        if window_batches == accumulation_steps or batch_number == len(train_loader):
            denominator = window_batches * (mean_batch_pixels if objective == "pixel" else 1)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(denominator)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip or float('inf'), error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_batches = 0
        total_loss += loss.item() * target.numel()
        count += target.numel()
        if stats is not None:
            targets64 = target.detach().double()
            target_sum += targets64.sum().item()
            target_square_sum += targets64.square().sum().item()
    if not count:
        raise ValueError("Training loader is empty")
    if stats is not None:
        target_variation = max(0.0, target_square_sum - target_sum ** 2 / count)
        stats.update(gradient_norm_mean=float(np.mean(norms)), gradient_norm_max=float(np.max(norms)),
                     optimizer_steps=len(norms), train_pixels=count,
                     train_rmse=float(np.sqrt(total_loss / count)),
                     train_r2=1.0 - total_loss / target_variation if target_variation > 0 else None)
    return total_loss / count


@torch.no_grad()
def model_validation(model, val_loader, criterion, device, BEST_R2_SCORE=-float("inf"), export_attention=False):
    model.eval()
    total_loss, count = 0.0, 0
    frames = []
    for batch in val_loader:
        target = batch["target"].to(device)
        predictions, weights = forward_batch(model, batch, device, return_attention=True)
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
        if export_attention and weights is not None:
            for step in range(weights.size(1)):
                frames[-1][f'attention_{step}'] = weights[:, step].cpu().numpy()
    if not count:
        raise ValueError("Validation loader is empty")
    frame = pd.concat(frames, ignore_index=True)
    score = float(r2_score(frame.target, frame.prediction)) if count > 1 and np.var(frame.target) > 0 else float("nan")
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
    parser.add_argument("--model", choices=("gnn_lstm", "lstm", "3d_gnn_lstm", "3d_lstm"), default="gnn_lstm")
    parser.add_argument("--cnn-hidden-dim", type=int, default=64, help="3D CNN output channels")
    parser.add_argument("--cnn-kernel-size", type=int, default=5, help="Odd spatial neighborhood width for 3D models")
    parser.add_argument("--cnn-temporal-kernel", type=int, default=1, help="Odd temporal kernel; 1 preserves independent time steps before LSTM")
    parser.add_argument("--cnn-max-grid-cells", type=int, default=65536, help="Maximum field bounding-grid area for 3D CNN allocation")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--aggregation", choices=("mean", "directional"), default="mean")
    parser.add_argument("--temporal-pool", choices=("last", "attention"), default="last")
    parser.add_argument("--attention-mask", action="store_true")
    parser.add_argument("--export-attention", action="store_true")
    parser.add_argument("--export-training-predictions", action="store_true",
                        help="Save training-pixel predictions from each fold's selected checkpoint (extra inference pass)")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--append-mask", action="store_true")
    parser.add_argument("--missing-value", type=float, help="Optional finite NetCDF missing sentinel; NaN/inf always missing")
    parser.add_argument("--cloud-mask", action="store_true")
    parser.add_argument("--scl-exclude", nargs="+", type=int, choices=range(12), default=list(DEFAULT_SCL_EXCLUDE))
    parser.add_argument("--diagnostics-only", action="store_true")
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
    parser.add_argument("--objective", choices=("graph_batch", "pixel"), default="graph_batch")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Graph batches per update; 0 accumulates whole epoch")
    parser.add_argument("--gradient-clip", type=float, default=5.0, help="Maximum norm; 0 disables clipping")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("none", "step", "plateau"), default="none")
    parser.add_argument("--scheduler-patience", type=int, default=5, help="StepLR interval or plateau patience")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--inner-val-fraction", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--splits", type=Path, help="Existing field-to-fold JSON manifest")
    parser.add_argument("--plots", action="store_true")
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
    if args.accumulation_steps < 0 or args.gradient_clip < 0 or args.weight_decay < 0:
        parser.error("accumulation-steps, gradient-clip and weight-decay must be nonnegative")
    if not 0 <= args.inner_val_fraction < 1 or args.early_stopping_patience < 0 or args.min_delta < 0:
        parser.error("Invalid inner validation or early stopping settings")
    if args.scheduler_patience < 1 or not 0 < args.lr_factor < 1:
        parser.error("scheduler-patience must be positive and 0 < lr-factor < 1")
    if (args.early_stopping_patience or args.scheduler == "plateau") and not args.inner_val_fraction:
        parser.error("Early stopping and plateau scheduling require --inner-val-fraction > 0")
    if args.cnn_hidden_dim < 1 or args.cnn_max_grid_cells < 1:
        parser.error("cnn-hidden-dim and cnn-max-grid-cells must be positive")
    if any(k < 1 or k % 2 == 0 for k in (args.cnn_kernel_size, args.cnn_temporal_kernel)):
        parser.error("CNN kernel sizes must be positive odd integers")
    if args.model in ('lstm', '3d_lstm') and (args.residual or args.aggregation != 'mean'):
        parser.error("Residual/directional options require a GNN model")
    if (args.attention_mask or args.export_attention) and args.temporal_pool != 'attention':
        parser.error("Attention masks/exports require --temporal-pool attention")
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
                           country=args.country, time_steps=args.time_steps,
                           cloud_mask=args.cloud_mask, scl_exclude=args.scl_exclude)
    # A run has its own directory so checkpoints and metrics cannot be overwritten accidentally.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['representation'] = 'growing_season_mean' if args.data.is_dir() else 'supplied_netcdf'
    config['actual_time_steps'] = dataset.sizes['time_step']
    config['input_fingerprint'] = input_fingerprint(dataset)
    write_json(args.output_dir / "config.json", config)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=SCRIPT_DIR, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    write_json(args.output_dir / 'environment.json', dict(python=platform.python_version(),
        torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__,
        cuda=torch.version.cuda, device=str(device), git_revision=revision, git_dirty=dirty,
        packages={name: version(name) for name in ('torch', 'numpy', 'pandas', 'xarray', 'scikit-learn', 'rasterio')},
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SCRIPT_DIR.glob('*.py')}))
    write_json(args.output_dir / "data_diagnostics.json", data_diagnostics(
        dataset, SpectralTransform(missing_value=args.missing_value)))
    if args.diagnostics_only:
        return
    splits = (splits_from_manifest(dataset, json.loads(args.splits.read_text()), args.folds)
              if args.splits else split_dataset_by_group(dataset, args.folds))
    write_json(args.output_dir / 'splits.json', split_manifest(dataset, splits))
    predictions, history = [], []
    for fold, (outer_train_indices, val_indices) in enumerate(splits):
        # Independent per-fold seeds make later folds insensitive to early stopping.
        random.seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        torch.manual_seed(args.seed + fold)
        train_indices, inner_indices = (inner_field_split(dataset, outer_train_indices, args.inner_val_fraction, args.seed + fold)
                                        if args.inner_val_fraction else (outer_train_indices, []))
        transform = SpectralTransform(normalize=args.normalize, append_mask=args.append_mask,
                                      missing_value=args.missing_value)
        if args.normalize:
            transform.fit(dataset['sample'].transpose('index', 'time_step', 'band').values,
                          dataset.index.to_index().get_indexer(train_indices))
        train_dataset = YieldSATDataset(dataset, indices=train_indices, band=None, connectivity=args.connectivity, transform=transform)
        val_dataset = YieldSATDataset(dataset, indices=val_indices, band=None, connectivity=args.connectivity, transform=transform)
        inner_dataset = (YieldSATDataset(dataset, indices=inner_indices, band=None,
                         connectivity=args.connectivity, transform=transform) if inner_indices else None)
        write_json(args.output_dir / f"fold_{fold}_diagnostics.json", {
            "train_target": distribution(train_dataset.targets), "outer_target": distribution(val_dataset.targets),
            "inner_target": distribution(inner_dataset.targets) if inner_dataset else None,
            "train_field_sizes": distribution([len(p) for p, _ in train_dataset.graphs]),
            "preprocessing": transform.state_dict()})
        loader_options = dict(
            batch_size=args.batch_size, collate_fn=collate_graphs,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        train_loader = DataLoader(
            train_dataset, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + fold), **loader_options,
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
        inner_loader = DataLoader(inner_dataset, shuffle=False, **loader_options) if inner_dataset else None
        model_config = dict(
            input_dim=train_dataset.num_features, hidden_dim=args.hidden_dim,
            num_layers=args.lstm_layers, gnn_hidden_dim=args.gnn_hidden_dim,
            gnn_layers=args.gnn_layers, dropout=args.dropout, model_type=args.model,
            residual=args.residual, aggregation=args.aggregation,
            temporal_pool=args.temporal_pool, attention_mask=args.attention_mask,
            cnn_hidden_dim=args.cnn_hidden_dim, cnn_kernel_size=args.cnn_kernel_size,
            cnn_temporal_kernel=args.cnn_temporal_kernel, cnn_max_grid_cells=args.cnn_max_grid_cells,
        )
        model = GNN_LSTM_Model(**model_config).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = None
        if args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_patience, gamma=args.lr_factor)
        elif args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, factor=args.lr_factor)
        best_loss, stale, selected_epoch, selected_state = float('inf'), 0, args.epochs, None
        for epoch in range(1, args.epochs + 1):
            stats = {}
            lr = optimizer.param_groups[0]['lr']
            train_mse = model_training(model, train_loader, criterion, optimizer, device,
                args.objective, args.accumulation_steps, args.gradient_clip, stats)
            inner_mse, inner_r2 = None, None
            if inner_loader is not None:
                inner_frame, _ = model_validation(model, inner_loader, criterion, device)
                inner_mse, inner_r2 = inner_frame.attrs['mse'], inner_frame.attrs['r2']
                if inner_mse < best_loss - args.min_delta:
                    best_loss, stale, selected_epoch = inner_mse, 0, epoch
                    selected_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
            history.append(dict(cv_fold=fold, epoch=epoch, train_mse=train_mse,
                inner_val_mse=inner_mse, inner_val_r2=inner_r2, learning_rate=lr, **stats))
            pd.DataFrame(history).to_csv(args.output_dir / 'history.csv', index=False)
            print(f"Fold {fold + 1}/{args.folds}, epoch {epoch}/{args.epochs}: "
                  f"train MSE={train_mse:.4f}, inner MSE={inner_mse}, lr={lr:.3g}", flush=True)
            if scheduler is not None:
                scheduler.step(inner_mse) if args.scheduler == 'plateau' else scheduler.step()
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                break
        if selected_state is not None:
            model.load_state_dict(selected_state)
        # Outer data are evaluated once, after selection; never used by scheduler/stopping.
        val_frame, _ = model_validation(model, val_loader, criterion, device, export_attention=args.export_attention)
        val_frame["cv_fold"] = fold
        val_frame["epoch"] = selected_epoch
        val_frame['train_mean_prediction'] = float(train_dataset.targets.mean())
        predictions.append(val_frame)
        val_frame.to_csv(args.output_dir / f"fold_{fold}_predictions.csv", index=False)
        torch.save({
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": model_config, "bands": list(args.bands),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "preprocessing": transform.state_dict(), "cloud_mask": args.cloud_mask,
            "scl_exclude": args.scl_exclude,
            "connectivity": args.connectivity, "crop": args.crop,
            "country": args.country, "time_steps": dataset.sizes['time_step'],
            "temporal_aggregation": config['representation'], "run_config": config,
            "fill_value": train_dataset.fill_value, "cv_fold": fold, "epoch": selected_epoch,
            "selection": "inner_mse" if inner_loader is not None else "fixed_final_epoch",
            "outer_train_indices": outer_train_indices, "inner_indices": inner_indices,
            "train_indices": train_indices, "val_indices": val_indices,
        }, args.output_dir / f"fold_{fold}.pt")
        if args.export_training_predictions:
            training_export_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
            training_frame, _ = model_validation(model, training_export_loader, criterion, device,
                                                 export_attention=args.export_attention)
            training_frame['cv_fold'] = fold
            training_frame['epoch'] = selected_epoch
            training_frame['split'] = 'train'
            training_frame['residual'] = training_frame.prediction - training_frame.target
            training_frame.to_csv(args.output_dir / f'fold_{fold}_training_predictions.csv', index=False)
            del training_frame, training_export_loader
    results = pd.concat(predictions, ignore_index=True)
    results.to_csv(args.output_dir / "cv_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    from eval import save_evaluation
    metrics = save_evaluation(results, args.output_dir, plots=args.plots, history=pd.DataFrame(history))
    print(json.dumps({level: {key: value for key, value in values.items() if not isinstance(value, dict)}
                      for level, values in metrics.items()}, indent=2))
    return results


def input_fingerprint(dataset):
    """Hash selected input/target/grid values in chunks, not the multi-GB source file."""
    digest = hashlib.sha256()
    for key in ('index', 'band', 'time_step', 'field_shared_name', 'year', 'row', 'col'):
        digest.update(json.dumps(dataset[key].values.astype(str).tolist()).encode('utf-8'))
    digest.update(np.asarray(dataset.target.values, dtype='<f4').tobytes())
    samples = dataset['sample'].transpose('index', 'time_step', 'band').values
    for start in range(0, len(samples), 4096):
        digest.update(np.asarray(samples[start:start + 4096], dtype='<f4').tobytes())
    return digest.hexdigest()


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GNN_LSTM_Model(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    transform = SpectralTransform(**checkpoint.get('preprocessing', {'fill_value': checkpoint.get('fill_value', -1)}))
    return model, transform, checkpoint


if __name__ == "__main__":
    run_cross_validation(parse_args())
