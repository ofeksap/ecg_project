#!/usr/bin/env python3
"""Validate the processed PTB-XL segment dataset before ECG-FM fine-tuning.

Checks that outputs from scripts 01-03 exist, are mutually consistent, and match
the expected fairseq-compatible layout:

- ``metadata/samples.csv`` aligns with ``labels/*`` and ``manifests/*.tsv``
- Waveform ``.mat`` files exist under ``waveforms/{train,valid,test}/``
- Label arrays are binary, row-aligned, and split counts look reasonable

Writes a JSON summary to ``metadata/dataset_check_report.json``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import load_paths  # noqa: E402

SPLIT_FOLDS = {
    "train": set(range(1, 9)),
    "valid": {9},
    "test": {10},
}


def fail(msg: str) -> None:
    """Print a failure message and exit with code 1."""
    print(f"[FAIL] {msg}")
    sys.exit(1)


def warn(msg: str) -> None:
    """Print a non-fatal warning."""
    print(f"[WARN] {msg}")


def ok(msg: str) -> None:
    """Print a successful check message."""
    print(f"[OK] {msg}")


def assign_split(strat_fold: int) -> str:
    """Map a PTB-XL stratified fold to train, valid, or test."""
    for split_name, folds in SPLIT_FOLDS.items():
        if strat_fold in folds:
            return split_name
    raise ValueError(f"Unexpected strat_fold: {strat_fold}")


def normalize_samples(samples: pd.DataFrame, segment_length: int) -> pd.DataFrame:
    """Derive ``split``, ``relpath``, and ``length`` columns when absent."""
    out = samples.copy()

    if "split" not in out.columns:
        if "strat_fold" not in out.columns:
            fail("samples.csv must contain either 'split' or 'strat_fold'")
        out["split"] = out["strat_fold"].map(assign_split)

    if "relpath" not in out.columns:
        if "mat_path" not in out.columns:
            fail("samples.csv must contain either 'relpath' or 'mat_path'")
        mat_names = out["mat_path"].map(lambda path: Path(path).name)
        out["relpath"] = out["split"] + "/" + mat_names

    if "length" not in out.columns:
        out["length"] = segment_length

    return out


def parse_pos_weight(path: Path) -> np.ndarray:
    """Parse ``pos_weight.txt`` into a float vector."""
    text = path.read_text().strip()

    try:
        values = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        text = text.replace("[", "").replace("]", "")
        values = [float(x) for x in text.replace(",", " ").split()]

    return np.asarray(values, dtype=np.float32)


def read_manifest(path: Path) -> tuple[Path, pd.DataFrame]:
    """Read a fairseq manifest TSV into its root path and entry table."""
    lines = path.read_text().splitlines()
    if not lines:
        fail(f"Manifest is empty: {path}")

    root = Path(lines[0])
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            fail(f"Bad manifest line in {path}: {line}")

        relpath, length = parts
        rows.append({"relpath": relpath, "length": int(length)})

    return root, pd.DataFrame(rows)


def check_required_files(processed_root: Path) -> None:
    """Ensure all expected processed dataset files exist."""
    required = [
        processed_root / "metadata" / "samples.csv",
        processed_root / "labels" / "record_labels.csv",
        processed_root / "labels" / "labels.csv",
        processed_root / "labels" / "label_def.csv",
        processed_root / "labels" / "y.npy",
        processed_root / "labels" / "pos_weight.txt",
        processed_root / "manifests" / "train.tsv",
        processed_root / "manifests" / "valid.tsv",
        processed_root / "manifests" / "test.tsv",
    ]

    for path in required:
        if not path.exists():
            fail(f"Missing required file: {path}")

    ok("All required metadata/label/manifest files exist.")


def check_samples(samples: pd.DataFrame) -> None:
    """Validate segment metadata ordering and split assignments."""
    required_cols = {"idx", "ecg_id", "segment_idx", "split", "relpath", "length"}
    missing = required_cols - set(samples.columns)
    if missing:
        fail(f"samples.csv is missing columns: {missing}")

    if samples["idx"].duplicated().any():
        fail("samples.csv has duplicated idx values.")

    expected_idx = np.arange(len(samples))
    if not np.array_equal(samples["idx"].to_numpy(), expected_idx):
        fail("samples.csv idx must be sequential: 0, 1, 2, ...")

    bad_splits = set(samples["split"].unique()) - {"train", "valid", "test"}
    if bad_splits:
        fail(f"samples.csv contains invalid split values: {bad_splits}")

    ok(f"samples.csv is valid: {len(samples)} samples.")


def check_labels(
    samples: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_def: pd.DataFrame,
    y: np.ndarray,
) -> list[str]:
    """Verify labels.csv, label_def.csv, and y.npy are aligned with samples."""
    if "name" not in label_def.columns:
        fail("label_def.csv must contain a 'name' column.")

    label_names = label_def["name"].astype(str).tolist()
    if not label_names:
        fail("No labels found in label_def.csv.")

    if y.ndim != 2:
        fail(f"y.npy must be 2D, got shape {y.shape}")
    if y.shape[0] != len(samples):
        fail(f"y.npy rows ({y.shape[0]}) != samples.csv rows ({len(samples)})")
    if y.shape[1] != len(label_names):
        fail(f"y.npy columns ({y.shape[1]}) != number of labels ({len(label_names)})")

    for col in ["idx", "ecg_id", "segment_idx"]:
        if col not in labels_df.columns:
            fail(f"labels.csv missing required column: {col}")

    missing_label_cols = [label for label in label_names if label not in labels_df.columns]
    if missing_label_cols:
        fail(f"labels.csv missing label columns: {missing_label_cols}")

    if len(labels_df) != len(samples):
        fail(f"labels.csv rows ({len(labels_df)}) != samples.csv rows ({len(samples)})")
    if not np.array_equal(labels_df["idx"].to_numpy(), samples["idx"].to_numpy()):
        fail("labels.csv idx does not match samples.csv idx.")

    labels_matrix = labels_df[label_names].to_numpy(dtype=np.float32)
    if not np.array_equal(labels_matrix, y):
        fail("labels.csv label columns do not match y.npy.")
    if not np.all((y == 0) | (y == 1)):
        fail("y.npy should contain only binary 0/1 values.")

    ok(f"Labels are aligned: y.npy shape = {y.shape}")
    return label_names


def check_pos_weight(pos_weight: np.ndarray, label_names: list[str]) -> None:
    """Validate class weight vector length and values."""
    if len(pos_weight) != len(label_names):
        fail(f"pos_weight length ({len(pos_weight)}) != number of labels ({len(label_names)})")
    if not np.all(np.isfinite(pos_weight)):
        fail("pos_weight.txt contains NaN or infinity.")
    if np.any(pos_weight < 0):
        fail("pos_weight.txt contains negative values.")

    ok("pos_weight.txt is valid.")


def check_label_distribution(
    samples: pd.DataFrame,
    y: np.ndarray,
    label_names: list[str],
) -> tuple[dict[str, int], list[dict[str, int | str]]]:
    """Print per-split sample counts and per-label positive counts."""
    train_mask = samples["split"].eq("train").to_numpy()
    valid_mask = samples["split"].eq("valid").to_numpy()
    test_mask = samples["split"].eq("test").to_numpy()

    split_counts = {
        "train": int(train_mask.sum()),
        "valid": int(valid_mask.sum()),
        "test": int(test_mask.sum()),
    }

    print("\nSplit counts:")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")

    if split_counts["train"] == 0:
        fail("Train split has 0 samples.")
    if split_counts["valid"] == 0:
        warn("Valid split has 0 samples.")
    if split_counts["test"] == 0:
        warn("Test split has 0 samples.")

    y_train = y[train_mask]
    y_valid = y[valid_mask]
    y_test = y[test_mask]

    train_pos = y_train.sum(axis=0).astype(int)
    valid_pos = y_valid.sum(axis=0).astype(int) if len(y_valid) else np.zeros(len(label_names), dtype=int)
    test_pos = y_test.sum(axis=0).astype(int) if len(y_test) else np.zeros(len(label_names), dtype=int)

    print("\nLabel distribution:")
    dist_rows = []
    for i, label in enumerate(label_names):
        row = {
            "label": label,
            "train_pos": int(train_pos[i]),
            "valid_pos": int(valid_pos[i]),
            "test_pos": int(test_pos[i]),
        }
        dist_rows.append(row)
        print(
            f"  {label:15s} "
            f"train={row['train_pos']:5d} "
            f"valid={row['valid_pos']:5d} "
            f"test={row['test_pos']:5d}"
        )

        if train_pos[i] == 0:
            warn(f"Label '{label}' has 0 positive examples in train.")
        if split_counts["valid"] and valid_pos[i] == 0:
            warn(f"Label '{label}' has 0 positive examples in valid.")
        if split_counts["test"] and test_pos[i] == 0:
            warn(f"Label '{label}' has 0 positive examples in test.")

    ok("Label distribution checked.")
    return split_counts, dist_rows


def check_manifests(processed_root: Path, samples: pd.DataFrame) -> None:
    """Verify train/valid/test manifests match samples.csv."""
    waveforms_dir = processed_root / "waveforms"
    manifests_dir = processed_root / "manifests"
    total_manifest_rows = 0

    for split in ["train", "valid", "test"]:
        manifest_path = manifests_dir / f"{split}.tsv"
        manifest_root, manifest_df = read_manifest(manifest_path)

        expected_root = waveforms_dir.resolve()
        if manifest_root.resolve() != expected_root:
            fail(
                f"{split}.tsv root is wrong.\n"
                f"Expected: {expected_root}\n"
                f"Got:      {manifest_root.resolve()}"
            )

        expected_df = samples[samples["split"] == split].sort_values("idx")
        if len(manifest_df) != len(expected_df):
            fail(
                f"{split}.tsv has {len(manifest_df)} rows, "
                f"but samples.csv has {len(expected_df)} {split} samples."
            )

        if len(expected_df):
            if not np.array_equal(manifest_df["relpath"].to_numpy(), expected_df["relpath"].to_numpy()):
                fail(f"{split}.tsv relpath order does not match samples.csv idx order.")
            if not np.array_equal(manifest_df["length"].to_numpy(), expected_df["length"].to_numpy()):
                fail(f"{split}.tsv length column does not match samples.csv.")

        total_manifest_rows += len(manifest_df)
        ok(f"{split}.tsv is aligned with samples.csv: {len(manifest_df)} rows.")

    if total_manifest_rows != len(samples):
        fail(f"Total manifest rows ({total_manifest_rows}) != total samples ({len(samples)})")

    ok("All manifests are valid.")


def check_waveform_files(
    processed_root: Path,
    samples: pd.DataFrame,
    sample_rate: int,
    expected_leads: int,
    expected_length: int,
    deep: bool,
) -> None:
    """Ensure manifest-linked ``.mat`` files exist and contain valid waveforms."""
    waveforms_dir = processed_root / "waveforms"

    missing = [str(waveforms_dir / relpath) for relpath in samples["relpath"] if not (waveforms_dir / relpath).exists()]
    if missing:
        print("\nFirst missing files:")
        for path in missing[:10]:
            print(" ", path)
        fail(f"{len(missing)} waveform files are missing.")

    ok("All waveform files exist.")

    if deep:
        check_df = samples
        print("\nDeep-checking all .mat files...")
    else:
        n = min(50, len(samples))
        check_df = samples.sample(n=n, random_state=0)
        print(f"\nChecking a random subset of {n} .mat files...")

    for _, row in tqdm(check_df.iterrows(), total=len(check_df), desc="Checking .mat"):
        relpath = row["relpath"]
        path = waveforms_dir / relpath

        try:
            mat = scipy.io.loadmat(path)
        except (OSError, ValueError) as exc:
            fail(f"Could not read .mat file {path}: {exc}")

        for key in ["feats", "curr_sample_rate", "patient_id", "idx"]:
            if key not in mat:
                fail(f"{path} missing key: {key}")

        feats = mat["feats"]
        if feats.shape != (expected_leads, expected_length):
            fail(
                f"{path} has wrong feats shape {feats.shape}; "
                f"expected {(expected_leads, expected_length)}"
            )
        if not np.all(np.isfinite(feats)):
            fail(f"{path} contains NaN or infinity in feats.")

        mat_patient_id = int(np.ravel(mat["patient_id"])[0])
        if mat_patient_id != int(row["patient_id"]):
            fail(
                f"{path} has patient_id={mat_patient_id}, "
                f"but samples.csv expects patient_id={int(row['patient_id'])}"
            )

        mat_idx = int(np.ravel(mat["idx"])[0])
        if mat_idx != int(row["idx"]):
            fail(f"{path} has idx={mat_idx}, but samples.csv expects idx={int(row['idx'])}")

        fs = int(np.ravel(mat["curr_sample_rate"])[0])
        if fs != sample_rate:
            fail(f"{path} has curr_sample_rate={fs}, expected {sample_rate}")

    if deep:
        ok("All .mat files passed deep check.")
    else:
        ok("Random .mat subset passed check.")


def save_report(processed_root: Path, report: dict) -> None:
    """Write the validation summary as JSON under ``metadata/``."""
    out_dir = processed_root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_check_report.json"

    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    ok(f"Saved check report to {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Processed dataset root (defaults to processed_root in paths.yaml).",
    )
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--segment-seconds", type=int, default=5)
    parser.add_argument("--expected-leads", type=int, default=12)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Load and check every .mat file instead of a random subset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths()
    processed_root = args.processed_root or paths["processed_root"]

    if not processed_root.exists():
        fail(f"Processed dataset root does not exist: {processed_root}")

    expected_length = args.sample_rate * args.segment_seconds

    print("Checking dataset:")
    print("  processed_root:", processed_root)
    print("  expected shape:", (args.expected_leads, expected_length))
    print()

    check_required_files(processed_root)

    samples = pd.read_csv(processed_root / "metadata" / "samples.csv")
    samples = normalize_samples(samples, expected_length)

    labels_df = pd.read_csv(processed_root / "labels" / "labels.csv")
    label_def = pd.read_csv(processed_root / "labels" / "label_def.csv")
    y = np.load(processed_root / "labels" / "y.npy")
    pos_weight = parse_pos_weight(processed_root / "labels" / "pos_weight.txt")

    check_samples(samples)
    label_names = check_labels(samples, labels_df, label_def, y)
    check_pos_weight(pos_weight, label_names)
    split_counts, label_distribution = check_label_distribution(samples, y, label_names)
    check_manifests(processed_root, samples)
    check_waveform_files(
        processed_root=processed_root,
        samples=samples,
        sample_rate=args.sample_rate,
        expected_leads=args.expected_leads,
        expected_length=expected_length,
        deep=args.deep,
    )

    report = {
        "processed_root": str(processed_root),
        "num_samples": int(len(samples)),
        "num_labels": int(len(label_names)),
        "label_names": label_names,
        "split_counts": split_counts,
        "label_distribution": label_distribution,
        "sample_rate": args.sample_rate,
        "segment_seconds": args.segment_seconds,
        "expected_leads": args.expected_leads,
        "expected_length": expected_length,
    }
    save_report(processed_root, report)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
