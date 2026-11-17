"""Evaluate one out-of-fold prediction per pixel, optionally saving plots."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(targets, predictions):
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "r2": float(r2_score(targets, predictions)) if len(targets) > 1 else None,
        "count": len(targets),
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
    fields = predictions.groupby(["field_shared_name", "year"], as_index=False).agg(
        target=("target", "mean"), prediction=("prediction", "mean"),
    )
    return {
        "pixel": regression_metrics(predictions.target, predictions.prediction),
        "field_year": regression_metrics(fields.target, fields.prediction),
    }, fields


def save_plots(pixels, fields, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, frame in (("pixel", pixels), ("field_year", fields)):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].scatter(frame.target, frame.prediction, alpha=0.4, s=10)
        low = min(frame.target.min(), frame.prediction.min())
        high = max(frame.target.max(), frame.prediction.max())
        axes[0].plot([low, high], [low, high], color="black")
        axes[0].set(xlabel="Target (t/ha)", ylabel="Prediction (t/ha)", title=name)
        axes[1].hist(frame.target, bins=30, alpha=0.5, label="Target")
        axes[1].hist(frame.prediction, bins=30, alpha=0.5, label="Prediction")
        axes[1].set(xlabel="Yield (t/ha)", ylabel="Count")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_evaluation.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    pixels = pd.read_csv(args.predictions, dtype={"index": str})
    metrics, fields = evaluate_predictions(pixels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields.to_csv(args.output_dir / "field_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.plots:
        save_plots(pixels, fields, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
