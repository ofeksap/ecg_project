#!/usr/bin/env python3
"""Evaluate saved test predictions against held-out PTB-XL labels.

Loads ``test_logits.npy`` or ``test_predictions.npy`` from a predictions directory,
compares against the test split of ``labels/y.npy``, and writes metric tables.
Does not train or run inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import load_paths, load_test_ground_truth  # noqa: E402


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    if y_true.sum() == 0 and y_pred.sum() == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _micro_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    metrics = _binary_metrics(y_true.ravel(), y_pred.ravel())
    metrics["auroc"] = _safe_auroc(y_true.ravel(), y_score.ravel())
    metrics["auprc"] = _safe_auprc(y_true.ravel(), y_score.ravel())
    metrics["precision"] = float(precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
    metrics["recall"] = float(recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
    metrics["f1"] = float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
    return metrics


def load_prediction_arrays(predictions_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    logits_path = predictions_dir / "test_logits.npy"
    preds_path = predictions_dir / "test_predictions.npy"

    if logits_path.is_file():
        logits = np.load(logits_path)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.int8)
        return logits.astype(np.float32), preds

    if preds_path.is_file():
        preds = np.load(preds_path)
        return preds.astype(np.float32), preds.astype(np.int8)

    raise FileNotFoundError(
        f"Missing prediction files in {predictions_dir}. "
        "Expected test_logits.npy or test_predictions.npy."
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for i, label in enumerate(label_names):
        cls_true = y_true[:, i]
        cls_pred = y_pred[:, i]
        cls_score = y_score[:, i]
        bin_metrics = _binary_metrics(cls_true, cls_pred)
        rows.append({
            "label": label,
            "support": int(cls_true.sum()),
            "auroc": _safe_auroc(cls_true, cls_score),
            "auprc": _safe_auprc(cls_true, cls_score),
            **bin_metrics,
        })

    per_label = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {
            "average": "macro",
            "auroc": _nanmean(per_label["auroc"].tolist()),
            "auprc": _nanmean(per_label["auprc"].tolist()),
            "precision": _nanmean(per_label["precision"].tolist()),
            "recall": _nanmean(per_label["recall"].tolist()),
            "f1": _nanmean(per_label["f1"].tolist()),
        },
        {
            "average": "micro",
            **_micro_metrics(y_true, y_pred, y_score),
        },
    ])

    report = {
        "num_test_samples": int(y_true.shape[0]),
        "num_labels": int(len(label_names)),
        "label_names": label_names,
        "per_label": per_label.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    return per_label, summary, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing test_logits.npy or test_predictions.npy",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Labels directory (defaults to paths.yaml labels_dir).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Metadata directory (defaults to paths.yaml metadata_dir).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Output directory for metrics tables (defaults to <predictions-dir>/metrics).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths()

    labels_dir = args.labels_dir or paths["labels_dir"]
    metadata_dir = args.metadata_dir or paths["metadata_dir"]
    metrics_dir = args.metrics_dir or (args.predictions_dir / "metrics")

    y_true, _, label_names = load_test_ground_truth(labels_dir, metadata_dir)
    y_score, y_pred = load_prediction_arrays(args.predictions_dir)

    if y_pred.shape != y_true.shape:
        print(
            f"Shape mismatch: predictions {y_pred.shape} vs ground truth {y_true.shape}"
        )
        return 1

    per_label, summary, report = compute_metrics(y_true, y_pred, y_score, label_names)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_label.to_csv(metrics_dir / "metrics_per_label.csv", index=False)
    summary.to_csv(metrics_dir / "metrics_summary.csv", index=False)
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Predictions dir:", args.predictions_dir)
    print("Saved metrics to:", metrics_dir)
    print("\nSummary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
