"""Evaluate one out-of-fold prediction per pixel, optionally saving plots."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from diagnostics import distribution, write_json


def regression_metrics(targets, predictions):
    targets, predictions = np.asarray(targets, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    mse = float(mean_squared_error(targets, predictions))
    target_var, prediction_var = float(np.var(targets)), float(np.var(predictions))
    return {
        "mse": mse, "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(targets, predictions)) if len(targets) > 1 and target_var > 0 else None,
        "pearson": float(np.corrcoef(targets, predictions)[0, 1]) if target_var > 0 and prediction_var > 0 else None,
        "prediction_target_variance_ratio": prediction_var / target_var if target_var > 0 else None,
        "bias": float(np.mean(predictions - targets)),
        "count": len(targets),
        "target_distribution": distribution(targets), "prediction_distribution": distribution(predictions),
    }


def evaluate_predictions(predictions):
    required = {"index", "target", "prediction", "field_shared_name", "year"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty or predictions["index"].isna().any():
        raise ValueError("Predictions must contain nonmissing pixel indices")
    if predictions["index"].duplicated().any():
        raise ValueError("Expected one out-of-fold prediction per pixel; select one epoch first")
    if predictions[["field_shared_name", "year"]].isna().any().any():
        raise ValueError("Field and year metadata must not be missing")
    if not np.isfinite(predictions[["target", "prediction"]].to_numpy()).all():
        raise ValueError("Targets and predictions must be finite")
    predictions = predictions.astype({'target': 'float64', 'prediction': 'float64'})
    fields = predictions.groupby(["field_shared_name", "year"], as_index=False).agg(
        target=("target", "mean"), prediction=("prediction", "mean"),
    )
    if 'cv_fold' in predictions:
        if predictions.cv_fold.isna().any() or (predictions.groupby('field_shared_name').cv_fold.nunique() != 1).any():
            raise ValueError("Each field must belong to exactly one outer fold")
        fields = fields.merge(predictions[['field_shared_name', 'cv_fold']].drop_duplicates(), on='field_shared_name')
    return {
        "pixel": regression_metrics(predictions.target, predictions.prediction),
        "field_year": regression_metrics(fields.target, fields.prediction),
    }, fields


def save_plots(pixels, fields, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, frame in (("pixel", pixels), ("field_year", fields)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].scatter(frame.target, frame.prediction, alpha=0.4, s=10)
        low = min(frame.target.min(), frame.prediction.min())
        high = max(frame.target.max(), frame.prediction.max())
        axes[0].plot([low, high], [low, high], color="black")
        axes[0].set(xlabel="Target (t/ha)", ylabel="Prediction (t/ha)", title=name)
        axes[1].hist(frame.target, bins=30, alpha=0.5, label="Target")
        axes[1].hist(frame.prediction, bins=30, alpha=0.5, label="Prediction")
        axes[1].set(xlabel="Yield (t/ha)", ylabel="Count")
        axes[1].legend()
        axes[2].hist(frame.prediction - frame.target, bins=40)
        axes[2].axvline(0, color='black')
        axes[2].set(xlabel='Prediction - target (t/ha)', ylabel='Count', title='Residuals')
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_evaluation.png", dpi=150)
        plt.close(fig)


def save_training_curves(history, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for fold, frame in history.groupby('cv_fold'):
        axes[0].plot(frame.epoch, frame.train_mse, label=f'Fold {fold} train')
        if 'inner_val_mse' in frame and frame.inner_val_mse.notna().any():
            axes[0].plot(frame.epoch, frame.inner_val_mse, linestyle='--', label=f'Fold {fold} inner')
        if 'learning_rate' in frame:
            axes[1].plot(frame.epoch, frame.learning_rate, label=f'Fold {fold}')
    axes[0].set(xlabel='Epoch', ylabel='MSE (t/ha)^2')
    axes[1].set(xlabel='Epoch', ylabel='Learning rate')
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'training_curves.png', dpi=150)
    plt.close(fig)


def save_evaluation(pixels, output_dir, plots=False, history=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, fields = evaluate_predictions(pixels)
    fields.to_csv(output_dir / 'field_predictions.csv', index=False)
    write_json(output_dir / 'metrics.json', metrics)
    if 'cv_fold' in pixels:
        rows = []
        for fold, frame in pixels.groupby('cv_fold'):
            fold_metrics, _ = evaluate_predictions(frame)
            for level, values in fold_metrics.items():
                rows.append(dict(cv_fold=int(fold), level=level,
                    **{k: v for k, v in values.items() if not isinstance(v, dict)}))
        per_fold = pd.DataFrame(rows)
        per_fold.to_csv(output_dir / 'per_fold_metrics.csv', index=False)
        summary = {}
        for level, frame in per_fold.groupby('level'):
            summary[level] = {}
            for name in ('rmse', 'mse', 'r2', 'pearson'):
                values = frame[name].dropna()
                summary[level][name] = dict(mean=float(values.mean()) if len(values) else None,
                    std=float(values.std(ddof=1)) if len(values) > 1 else None, defined_folds=len(values))
        write_json(output_dir / 'fold_summary.json', summary)
    if 'train_mean_prediction' in pixels:
        baseline = pixels.copy()
        baseline['prediction'] = baseline.train_mean_prediction
        write_json(output_dir / 'train_mean_baseline_metrics.json', evaluate_predictions(baseline)[0])
    if plots:
        save_plots(pixels, fields, output_dir)
        if history is not None:
            save_training_curves(history, output_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--history", type=Path, help="Defaults to history.csv beside predictions when present")
    args = parser.parse_args()
    pixels = pd.read_csv(args.predictions, dtype={"index": str})
    history_path = args.history or args.predictions.parent / 'history.csv'
    history = pd.read_csv(history_path) if history_path.exists() else None
    metrics = save_evaluation(pixels, args.output_dir, plots=args.plots, history=history)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
