#!/usr/bin/env python3
"""Train a randomly initialized fairseq ECG transformer baseline on PTB-XL.

Uses the same ``ecg_transformer_classifier`` architecture and data splits as the
ECG-FM fine-tuning scripts, but starts from random weights
(``model.no_pretrained_weights=true``). After training, runs test inference and
writes standardized prediction files for comparison with ECG-FM.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import (  # noqa: E402
    build_fairseq_train_cmd,
    load_paths,
    load_test_ground_truth,
    run_fairseq_test_inference,
    save_test_predictions,
    verify_test_logits,
)


def parse_pos_weight(path: Path) -> str:
    return path.read_text().strip().replace("\n", "").replace(" ", "")


def preflight(paths: dict[str, Path | str]) -> tuple[int, str]:
    fairseq_root = paths["fairseq_signals_root"]
    label_dir = paths["labels_dir"]
    manifest_dir = paths["manifest_dir"]

    if not fairseq_root.is_dir():
        raise FileNotFoundError(f"Missing fairseq-signals root: {fairseq_root}")

    required = [
        label_dir / "label_def.csv",
        label_dir / "y.npy",
        label_dir / "pos_weight.txt",
        manifest_dir / "train.tsv",
        manifest_dir / "valid.tsv",
        manifest_dir / "test.tsv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    for split in ("train", "valid", "test"):
        rows = sum(1 for _ in open(manifest_dir / f"{split}.tsv")) - 1
        if rows == 0:
            raise ValueError(f"{split}.tsv has no samples.")

    num_labels = len(pd.read_csv(label_dir / "label_def.csv"))
    pos_weight = parse_pos_weight(label_dir / "pos_weight.txt")
    return num_labels, pos_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Training output directory (defaults to baseline_output_dir in paths.yaml).",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory for test predictions (defaults to <output-dir>/predictions).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run test inference from an existing checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths()
    output_dir = args.output_dir or paths["baseline_output_dir"]
    predictions_dir = args.predictions_dir or (output_dir / "predictions")
    checkpoint = output_dir / "checkpoint_best.pt"

    num_labels, pos_weight = preflight(paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    fairseq_root = paths["fairseq_signals_root"]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{fairseq_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    if not args.skip_train:
        cmd = build_fairseq_train_cmd(
            fairseq_root,
            paths["manifest_dir"],
            paths["labels_dir"],
            output_dir,
            num_labels,
            pos_weight,
            model_path=None,
            no_pretrained_weights=True,
        )
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    if not checkpoint.is_file():
        print(f"Missing checkpoint: {checkpoint}")
        return 1

    _, test_meta, label_names = load_test_ground_truth(
        paths["labels_dir"],
        paths["metadata_dir"],
    )
    logits = run_fairseq_test_inference(
        checkpoint=checkpoint,
        manifest_dir=paths["manifest_dir"],
        fairseq_signals_root=fairseq_root,
        batch_size=args.batch_size,
        num_workers=0,
        device=args.device,
    )
    verify_test_logits(logits, test_meta, label_names)
    save_test_predictions(predictions_dir, logits, test_meta, label_names)

    print("Saved test predictions to:", predictions_dir)
    print("Logits shape:", logits.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
