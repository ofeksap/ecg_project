#!/usr/bin/env python3
"""Evaluate saved test predictions against held-out PTB-XL labels.

Loads ``test_logits.npy`` or ``test_predictions.npy`` from a predictions directory,
compares against the test split of ``labels/y.npy``, and writes metric tables.
Also writes per-segment and per-record mismatch tables for comparing split methods.

Outputs (under ``<predictions-dir>/metrics/``):

    metrics_per_label.csv, metrics_summary.csv, metrics.json
    per_segment.csv       One row per test segment with match / mismatch detail
    per_record.csv        One row per ecg_id with segment breakdown and record match
    mismatch_summary.json Aggregate mismatch rates (segment + record level)

When ``--aggregate-records mean``, pool-level record metrics are additionally
written under ``metrics/record/``.
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

from ecg_common import (  # noqa: E402
    aggregate_logits_by_record,
    load_paths,
    load_test_ground_truth,
)


def _active_label_set(row: np.ndarray, label_names: list[str]) -> list[str]:
    return [name for name, value in zip(label_names, row) if int(value) == 1]


def _label_set_str(labels: list[str]) -> str:
    return "|".join(labels) if labels else ""


def _load_segment_predictions_csv(
    predictions_dir: Path,
    label_names: list[str],
) -> tuple[np.ndarray | None, np.ndarray]:
    """Load segment predictions from ``test_predictions.csv`` if npy files are absent."""
    csv_path = predictions_dir / "test_predictions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing prediction files in {predictions_dir}. "
            "Expected test_logits.npy, test_predictions.npy, or test_predictions.csv."
        )

    pred_df = pd.read_csv(csv_path)
    missing = [name for name in label_names if name not in pred_df.columns]
    if missing:
        raise ValueError(f"test_predictions.csv missing label columns: {missing}")

    preds = pred_df[label_names].to_numpy(dtype=np.int8)
    logits_path = predictions_dir / "test_logits.npy"
    logits = np.load(logits_path).astype(np.float32) if logits_path.is_file() else None
    return logits, preds


def load_segment_prediction_arrays(
    predictions_dir: Path,
    label_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (y_score, y_pred, logits_or_none) for the test segment split."""
    logits_path = predictions_dir / "test_logits.npy"
    preds_path = predictions_dir / "test_predictions.npy"

    logits: np.ndarray | None = None
    if logits_path.is_file():
        logits = np.load(logits_path).astype(np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.int8)
        scores = probs.astype(np.float32)
    elif preds_path.is_file():
        preds = np.load(preds_path).astype(np.int8)
        scores = preds.astype(np.float32)
    else:
        logits, preds = _load_segment_predictions_csv(predictions_dir, label_names)
        scores = preds.astype(np.float32)

    return scores, preds, logits


def compute_per_segment_mismatches(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """Build a per-segment table with exact-match and label-level mismatch detail."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    segments_per_record = meta.groupby("ecg_id").size().to_dict()
    rows: list[dict[str, object]] = []

    for row_idx in range(y_true.shape[0]):
        true_row = y_true[row_idx]
        pred_row = y_pred[row_idx]
        diff = pred_row.astype(np.int8) - true_row.astype(np.int8)

        true_labels = _active_label_set(true_row, label_names)
        pred_labels = _active_label_set(pred_row, label_names)
        fp_labels = _active_label_set(np.clip(diff, 0, 1), label_names)
        fn_labels = _active_label_set(np.clip(-diff, 0, 1), label_names)

        meta_row = meta.iloc[row_idx]
        rows.append({
            "idx": int(meta_row["idx"]),
            "ecg_id": int(meta_row["ecg_id"]),
            "segment_idx": int(meta_row["segment_idx"]),
            "num_segments": int(segments_per_record[int(meta_row["ecg_id"])]),
            "true_labels": _label_set_str(true_labels),
            "pred_labels": _label_set_str(pred_labels),
            "exact_match": bool(np.array_equal(true_row, pred_row)),
            "hamming_error": float(np.mean(true_row != pred_row)),
            "n_false_positive": len(fp_labels),
            "n_false_negative": len(fn_labels),
            "false_positive_labels": _label_set_str(fp_labels),
            "false_negative_labels": _label_set_str(fn_labels),
        })

    return pd.DataFrame(rows)


def compute_per_record_mismatches(
    segment_df: pd.DataFrame,
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    *,
    logits: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize segment correctness and record-level match per ``ecg_id``."""
    score_array = logits if logits is not None else y_score
    if score_array is not None:
        record_score, record_meta = aggregate_logits_by_record(
            score_array, meta, aggregate="mean"
        )
        if logits is not None:
            record_probs = 1.0 / (1.0 + np.exp(-record_score))
            record_pred = (record_probs >= 0.5).astype(np.int8)
        else:
            record_pred = (record_score >= 0.5).astype(np.int8)
    else:
        grouped = (
            segment_df.sort_values(["ecg_id", "segment_idx"])
            .groupby("ecg_id", sort=True)
        )
        record_meta = (
            grouped.agg(idx=("idx", "min"), num_segments=("segment_idx", "count"))
            .reset_index()
        )
        record_pred = grouped[label_names].mean().ge(0.5).astype(np.int8).to_numpy()

    truth_by_ecg: dict[int, np.ndarray] = {}
    for ecg_id, group in meta.groupby("ecg_id", sort=True):
        first_row = int(group.index.to_numpy()[0])
        truth_by_ecg[int(ecg_id)] = y_true[first_row]

    rows: list[dict[str, object]] = []
    record_meta = record_meta.reset_index(drop=True)
    for rec_idx, rec_row in record_meta.iterrows():
        ecg_id = int(rec_row["ecg_id"])
        seg_rows = segment_df[segment_df["ecg_id"] == ecg_id].sort_values("segment_idx")
        seg_matches = seg_rows["exact_match"].tolist()
        seg_pred_sets = seg_rows["pred_labels"].tolist()

        true_vec = truth_by_ecg[ecg_id]
        pred_vec = record_pred[rec_idx]
        true_labels = _active_label_set(true_vec, label_names)
        pred_labels = _active_label_set(pred_vec, label_names)
        diff = pred_vec.astype(np.int8) - true_vec.astype(np.int8)

        rows.append({
            "ecg_id": ecg_id,
            "num_segments": int(rec_row["num_segments"]),
            "true_labels": _label_set_str(true_labels),
            "pred_labels": _label_set_str(pred_labels),
            "record_exact_match": bool(np.array_equal(pred_vec, true_vec)),
            "n_segments_correct": int(sum(seg_matches)),
            "n_segments_wrong": int(len(seg_matches) - sum(seg_matches)),
            "all_segments_match": bool(all(seg_matches)),
            "any_segment_wrong": bool(not all(seg_matches)),
            "segments_agree": len(set(seg_pred_sets)) == 1,
            "segment_exact_match": ",".join("1" if match else "0" for match in seg_matches),
            "segment_pred_labels": "|".join(seg_pred_sets),
            "n_false_positive": int(np.clip(diff, 0, 1).sum()),
            "n_false_negative": int(np.clip(-diff, 0, 1).sum()),
            "false_positive_labels": _label_set_str(
                _active_label_set(np.clip(diff, 0, 1), label_names)
            ),
            "false_negative_labels": _label_set_str(
                _active_label_set(np.clip(-diff, 0, 1), label_names)
            ),
        })

    return pd.DataFrame(rows)


def compute_mismatch_summary(
    segment_df: pd.DataFrame,
    record_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Aggregate mismatch rates useful for comparing split methods."""
    num_segments = len(segment_df)
    num_records = len(record_df)
    return {
        "num_test_segments": num_segments,
        "num_test_records": num_records,
        "segment_exact_match_rate": float(segment_df["exact_match"].mean()),
        "segment_hamming_error_mean": float(segment_df["hamming_error"].mean()),
        "record_exact_match_rate": float(record_df["record_exact_match"].mean()),
        "records_all_segments_match_rate": float(record_df["all_segments_match"].mean()),
        "records_any_segment_wrong_rate": float(record_df["any_segment_wrong"].mean()),
        "records_segment_prediction_disagreement_rate": float((~record_df["segments_agree"]).mean()),
        "records_partial_segment_match_rate": float(
            ((record_df["n_segments_correct"] > 0) & record_df["any_segment_wrong"]).mean()
        ),
    }


def write_mismatch_tables(
    metrics_dir: Path,
    segment_df: pd.DataFrame,
    record_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Write per-segment / per-record mismatch CSVs and return summary dict."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    segment_df.fillna("").to_csv(metrics_dir / "per_segment.csv", index=False)
    record_df.fillna("").to_csv(metrics_dir / "per_record.csv", index=False)

    summary = compute_mismatch_summary(segment_df, record_df)
    with open(metrics_dir / "mismatch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


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


def load_prediction_arrays(
    predictions_dir: Path,
    *,
    record_level: bool = False,
    segment_meta: pd.DataFrame | None = None,
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    prefix = "record_" if record_level else ""
    logits_path = predictions_dir / f"{prefix}test_logits.npy"
    preds_path = predictions_dir / f"{prefix}test_predictions.npy"

    if logits_path.is_file():
        logits = np.load(logits_path)
    elif record_level and (predictions_dir / "test_logits.npy").is_file():
        if segment_meta is None:
            raise ValueError("segment_meta required to aggregate segment logits at eval time")
        seg_logits = np.load(predictions_dir / "test_logits.npy")
        logits, _ = aggregate_logits_by_record(seg_logits, segment_meta, aggregate=aggregate)
    elif preds_path.is_file():
        preds = np.load(preds_path)
        return preds.astype(np.float32), preds.astype(np.int8)
    else:
        raise FileNotFoundError(
            f"Missing prediction files in {predictions_dir}. "
            f"Expected {prefix}test_logits.npy or {prefix}test_predictions.npy."
        )

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int8)
    return logits.astype(np.float32), preds


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
    parser.add_argument(
        "--aggregate-records",
        choices=("none", "mean"),
        default="none",
        help=(
            "Evaluate at 10 s record level by mean-aggregating segment predictions "
            "per ecg_id (uses record_test_*.npy if present)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths()

    labels_dir = args.labels_dir or paths["labels_dir"]
    metadata_dir = args.metadata_dir or paths["metadata_dir"]
    metrics_dir = args.metrics_dir or (args.predictions_dir / "metrics")

    # --- segment-level ground truth and predictions (always) ---
    y_true_seg, segment_meta, label_names = load_test_ground_truth(
        labels_dir, metadata_dir, record_level=False
    )
    y_score_seg, y_pred_seg, logits_seg = load_segment_prediction_arrays(
        args.predictions_dir, label_names
    )

    if y_pred_seg.shape != y_true_seg.shape:
        print(
            f"Shape mismatch: predictions {y_pred_seg.shape} vs ground truth {y_true_seg.shape}"
        )
        return 1

    per_segment = compute_per_segment_mismatches(
        segment_meta, y_true_seg, y_pred_seg, label_names
    )
    per_record = compute_per_record_mismatches(
        per_segment,
        segment_meta,
        y_true_seg,
        y_pred_seg,
        label_names,
        logits=logits_seg,
        y_score=y_score_seg,
    )
    mismatch_summary = write_mismatch_tables(metrics_dir, per_segment, per_record)

    # --- pool-level segment metrics ---
    per_label, summary, report = compute_metrics(
        y_true_seg, y_pred_seg, y_score_seg, label_names
    )
    report["eval_level"] = "segment"
    report["mismatch_summary"] = mismatch_summary

    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_label.to_csv(metrics_dir / "metrics_per_label.csv", index=False)
    summary.to_csv(metrics_dir / "metrics_summary.csv", index=False)
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Predictions dir:", args.predictions_dir)
    print("Saved metrics to:", metrics_dir)
    print("\nSegment summary:")
    print(summary.to_string(index=False))
    print("\nMismatch summary:")
    for key, value in mismatch_summary.items():
        if key.endswith("_rate") or key.endswith("_mean"):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # --- optional pool-level record metrics (mean aggregation) ---
    if args.aggregate_records == "mean":
        y_true_rec, record_meta, _ = load_test_ground_truth(
            labels_dir, metadata_dir, record_level=True
        )
        y_score_rec, y_pred_rec = load_prediction_arrays(
            args.predictions_dir,
            record_level=True,
            segment_meta=segment_meta,
            aggregate=args.aggregate_records,
        )

        if y_pred_rec.shape != y_true_rec.shape:
            print(
                f"Record shape mismatch: predictions {y_pred_rec.shape} "
                f"vs ground truth {y_true_rec.shape}"
            )
            return 1

        rec_per_label, rec_summary, rec_report = compute_metrics(
            y_true_rec, y_pred_rec, y_score_rec, label_names
        )
        rec_report["eval_level"] = "record"
        rec_report["aggregate"] = args.aggregate_records
        rec_report["mismatch_summary"] = mismatch_summary

        record_metrics_dir = metrics_dir / "record"
        record_metrics_dir.mkdir(parents=True, exist_ok=True)
        rec_per_label.to_csv(record_metrics_dir / "metrics_per_label.csv", index=False)
        rec_summary.to_csv(record_metrics_dir / "metrics_summary.csv", index=False)
        with open(record_metrics_dir / "metrics.json", "w") as f:
            json.dump(rec_report, f, indent=2)

        print("\nRecord-level summary (mean-aggregated pool metrics):")
        print(rec_summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
