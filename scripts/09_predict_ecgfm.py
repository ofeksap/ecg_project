#!/usr/bin/env python3
"""Run ECG-FM test-set inference and save standardized prediction files.

Loads a fairseq ``checkpoint_best.pt``, runs inference on ``manifests/test.tsv``,
and writes ``test_logits.npy``, ``test_predictions.npy``, and
``test_predictions.csv`` for downstream evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import (  # noqa: E402
    aggregate_logits_by_record,
    load_paths,
    load_test_ground_truth,
    run_fairseq_test_inference,
    save_test_predictions,
    verify_test_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = load_paths()
    default_checkpoint = paths["output_dir_pretrained"] / "checkpoint_best.pt"

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Path to fairseq checkpoint_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for prediction outputs (defaults to <checkpoint_dir>/predictions).",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Manifest directory (defaults to paths.yaml manifest_dir).",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Labels directory for metadata alignment (defaults to paths.yaml labels_dir).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Metadata directory (defaults to paths.yaml metadata_dir).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--aggregate-records",
        choices=("none", "mean"),
        default="none",
        help=(
            "After segment inference, mean-aggregate logits per ecg_id and save "
            "record_test_*.npy/csv for 10 s record-level evaluation."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths()

    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        print(f"Missing checkpoint: {checkpoint}")
        return 1

    manifest_dir = args.manifest_dir or paths["manifest_dir"]
    labels_dir = args.labels_dir or paths["labels_dir"]
    metadata_dir = args.metadata_dir or paths["metadata_dir"]
    output_dir = args.output_dir or (checkpoint.parent / "predictions")

    _, test_meta, label_names = load_test_ground_truth(labels_dir, metadata_dir)
    logits = run_fairseq_test_inference(
        checkpoint=checkpoint,
        manifest_dir=manifest_dir,
        fairseq_signals_root=paths["fairseq_signals_root"],
        batch_size=args.batch_size,
        num_workers=0,
        device=args.device,
    )
    verify_test_logits(logits, test_meta, label_names)
    save_test_predictions(output_dir, logits, test_meta, label_names)

    if args.aggregate_records == "mean":
        record_logits, record_meta = aggregate_logits_by_record(
            logits, test_meta, aggregate="mean"
        )
        save_test_predictions(
            output_dir,
            record_logits,
            record_meta,
            label_names,
            prefix="record_",
        )
        print("Record-level logits shape:", record_logits.shape)

    print("Checkpoint:", checkpoint)
    print("Saved test predictions to:", output_dir)
    print("Segment logits shape:", logits.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
