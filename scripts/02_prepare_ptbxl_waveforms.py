"""Split PTB-XL 10 s records into 5 s waveform segments for ECG-FM fine-tuning.

Reads record-level ``record_labels.csv`` from script 01, loads each 500 Hz WFDB
record, and writes two non-overlapping 5 s ``.mat`` files per record. Each
segment inherits the parent record's labels.

Outputs:
    waveforms/*.mat       fairseq-compatible mats with feats (12, 2500).
    metadata/samples.csv  Segment index, ecg_id, paths, and fold metadata.
    labels/labels.csv     Segment-level multi-hot labels.
    labels/y.npy          Float32 array aligned with labels.csv rows.
    labels/label_def.csv  Label names with train counts and pos_weight.
    labels/pos_weight.txt Train-set neg/pos ratio per label.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml"

SAMPLE_RATE = 500
SEGMENT_SEC = 5
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SEC
RECORD_SAMPLES = SEGMENT_SAMPLES * 2
N_LEADS = 12


def load_paths() -> dict[str, Path | str]:
    with PATHS_FILE.open() as f:
        raw = yaml.safe_load(f)
    return {
        key: Path(value) if key != "label_mode" else value
        for key, value in raw.items()
    }


def read_wfdb_record(record_path: Path) -> tuple[np.ndarray, int]:
    """Read a PTB-XL WFDB record and return signal in (leads, samples) layout."""
    hea_path = record_path.with_suffix(".hea")
    lines = hea_path.read_text().strip().splitlines()
    header = lines[0].split()
    n_sig = int(header[1])
    fs = int(header[2])
    sig_len = int(header[3])

    if n_sig != N_LEADS:
        raise ValueError(f"Expected {N_LEADS} leads, got {n_sig} for {record_path}")
    if fs != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {fs} for {record_path}")
    if sig_len != RECORD_SAMPLES:
        raise ValueError(
            f"Expected {RECORD_SAMPLES} samples, got {sig_len} for {record_path}"
        )

    gains: list[float] = []
    dat_file = None
    for line in lines[1 : 1 + n_sig]:
        parts = line.split()
        dat_file = parts[0]
        gain_match = re.match(r"([0-9.]+)", parts[2])
        if gain_match is None:
            raise ValueError(f"Could not parse gain from line: {line}")
        gains.append(float(gain_match.group(1)))

    raw = np.fromfile(record_path.parent / dat_file, dtype=np.int16)
    samples = len(raw) // n_sig
    signal = raw.reshape(samples, n_sig).T.astype(np.float32)
    for lead_idx, gain in enumerate(gains):
        signal[lead_idx] /= gain
    return signal, fs


def save_segment_mat(
    out_path: Path,
    feats: np.ndarray,
    patient_id: int,
    sample_rate: int,
    idx: int,
) -> None:
    scipy.io.savemat(
        out_path,
        {
            "idx": idx,
            "patient_id": patient_id,
            "curr_sample_rate": sample_rate,
            "feats": feats,
        },
    )


def write_segment_labels(
    labels_dir: Path,
    segment_rows: list[dict[str, int]],
    label_cols: list[str],
) -> None:
    labels_df = pd.DataFrame(segment_rows)
    train_mask = labels_df["strat_fold"].isin(range(1, 9)).to_numpy()
    labels_df = labels_df[["idx", "ecg_id", "segment_idx", *label_cols]]
    labels_df.to_csv(labels_dir / "labels.csv", index=False)

    y = labels_df[label_cols].to_numpy(dtype=np.float32)
    np.save(labels_dir / "y.npy", y)

    y_train = y[train_mask]
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pos_weight = neg / np.maximum(pos, 1)

    with open(labels_dir / "pos_weight.txt", "w") as f:
        f.write("[" + ",".join(f"{w:.6g}" for w in pos_weight) + "]\n")

    label_def = pd.DataFrame({
        "name": label_cols,
        "pos_count_train": pos.astype(int),
        "pos_percent_train": pos / y_train.shape[0],
        "pos_weight": pos_weight,
    })
    label_def.to_csv(labels_dir / "label_def.csv", index=False)

    print("Saved segment labels to:", labels_dir)
    print("y.npy shape:", y.shape)
    print("num labels:", len(label_cols))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records (for debugging).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths()
    raw_ptbxl = paths["raw_ptbxl"]
    labels_dir = paths["labels_dir"]
    metadata_dir = paths["metadata_dir"]
    waveform_dir = paths["waveform_dir"]

    record_labels_path = labels_dir / "record_labels.csv"
    if not record_labels_path.is_file():
        print(f"Missing record labels: {record_labels_path}")
        print("Run scripts/01_make_ptbxl_labels.py first.")
        return 1

    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    waveform_dir.mkdir(parents=True, exist_ok=True)

    record_labels = pd.read_csv(record_labels_path)
    if args.limit is not None:
        record_labels = record_labels.head(args.limit)

    label_cols = [
        col for col in record_labels.columns if col not in {"idx", "ecg_id"}
    ]
    meta = pd.read_csv(raw_ptbxl / "ptbxl_database.csv", index_col="ecg_id")

    sample_rows: list[dict] = []
    label_rows: list[dict] = []
    skipped = 0
    segment_idx_global = 0

    for record_idx, (_, row) in enumerate(record_labels.iterrows(), start=1):
        ecg_id = int(row["ecg_id"])
        record_meta = meta.loc[ecg_id]
        record_path = raw_ptbxl / record_meta.filename_hr

        try:
            signal, fs = read_wfdb_record(record_path)
        except (ValueError, FileNotFoundError, OSError) as exc:
            skipped += 1
            print(f"Skipping ecg_id={ecg_id}: {exc}")
            continue

        if np.isnan(signal).any():
            skipped += 1
            print(f"Skipping ecg_id={ecg_id}: NaN values detected")
            continue

        segments = (
            signal[:, :SEGMENT_SAMPLES],
            signal[:, SEGMENT_SAMPLES:RECORD_SAMPLES],
        )

        for segment_idx, feats in enumerate(segments):
            mat_name = f"{ecg_id:05d}_seg{segment_idx}.mat"
            mat_path = waveform_dir / mat_name
            save_segment_mat(
                mat_path,
                feats,
                patient_id=int(record_meta.patient_id),
                sample_rate=fs,
                idx=segment_idx_global,
            )

            sample_rows.append({
                "idx": segment_idx_global,
                "ecg_id": ecg_id,
                "segment_idx": segment_idx,
                "patient_id": int(record_meta.patient_id),
                "strat_fold": int(record_meta.strat_fold),
                "filename_hr": record_meta.filename_hr,
                "mat_path": str(mat_path.relative_to(paths["processed_root"])),
            })

            label_row = {
                "idx": segment_idx_global,
                "ecg_id": ecg_id,
                "segment_idx": segment_idx,
                "strat_fold": int(record_meta.strat_fold),
            }
            for label in label_cols:
                label_row[label] = int(row[label])
            label_rows.append(label_row)
            segment_idx_global += 1

        if record_idx % 1000 == 0:
            print(f"Processed {record_idx}/{len(record_labels)} records...")

    if not sample_rows:
        print("No segments were created.")
        return 1

    samples_df = pd.DataFrame(sample_rows)
    samples_df.to_csv(metadata_dir / "samples.csv", index=False)

    write_segment_labels(labels_dir, label_rows, label_cols)

    print("Saved waveforms to:", waveform_dir)
    print("Saved samples to:", metadata_dir / "samples.csv")
    print("num segments:", len(sample_rows))
    if skipped:
        print("skipped records:", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
