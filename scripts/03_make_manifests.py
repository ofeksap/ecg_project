"""Create fairseq-compatible manifest TSV files from samples.csv.

Reads ``metadata/samples.csv`` and assigns segments to train, valid, and test
using PTB-XL ``strat_fold`` (train: 1-8, valid: 9, test: 10). Creates split
subdirectories under ``waveform_dir`` with symlinks to the segment ``.mat``
files, then writes ``train.tsv``, ``valid.tsv``, and ``test.tsv`` to
``manifest_dir``.

Each manifest file looks like::

    /path/to/waveforms/root
    train/00001_seg0.mat\\t2500
    train/00001_seg1.mat\\t2500
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import load_paths  # noqa: E402

SEGMENT_SAMPLES = 2500

SPLIT_FOLDS = {
    "train": set(range(1, 9)),
    "valid": {9},
    "test": {10},
}


def assign_split(strat_fold: int) -> str:
    for split_name, folds in SPLIT_FOLDS.items():
        if strat_fold in folds:
            return split_name
    raise ValueError(f"Unexpected strat_fold: {strat_fold}")


def ensure_split_link(waveform_dir: Path, split: str, mat_name: str) -> None:
    split_dir = waveform_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    source = waveform_dir / mat_name
    link = split_dir / mat_name
    if not source.is_file():
        raise FileNotFoundError(f"Missing waveform file: {source}")

    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(source.resolve())


def write_manifest(manifest_path: Path, waveform_root: Path, entries: list[tuple[str, int]]) -> None:
    with manifest_path.open("w") as f:
        f.write(f"{waveform_root.resolve()}\n")
        for rel_path, num_samples in entries:
            f.write(f"{rel_path}\t{num_samples}\n")


def main() -> int:
    paths = load_paths()
    metadata_dir = paths["metadata_dir"]
    manifest_dir = paths["manifest_dir"]
    waveform_dir = paths["waveform_dir"]

    samples_path = metadata_dir / "samples.csv"
    if not samples_path.is_file():
        print(f"Missing samples metadata: {samples_path}")
        print("Run scripts/02_prepare_ptbxl_waveforms.py first.")
        return 1

    manifest_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(samples_path)
    required_cols = {"mat_path", "strat_fold"}
    missing = required_cols - set(samples.columns)
    if missing:
        raise ValueError(f"samples.csv is missing columns: {sorted(missing)}")

    samples["split"] = samples["strat_fold"].map(assign_split)
    samples["mat_name"] = samples["mat_path"].map(lambda p: Path(p).name)

    manifest_entries: dict[str, list[tuple[str, int]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for row in samples.itertuples(index=False):
        split = row.split
        mat_name = row.mat_name
        ensure_split_link(waveform_dir, split, mat_name)
        manifest_entries[split].append((f"{split}/{mat_name}", SEGMENT_SAMPLES))

    for split_name, entries in manifest_entries.items():
        manifest_path = manifest_dir / f"{split_name}.tsv"
        write_manifest(manifest_path, waveform_dir, entries)
        print(f"Wrote {manifest_path} ({len(entries)} segments)")

    print("Manifest root:", waveform_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
