"""Split PTB-XL 10 s records into 5 s waveform segments for ECG-FM fine-tuning.

Reads record-level ``record_labels.csv`` from script 01, loads each 500 Hz WFDB
record, and writes 5 s ``.mat`` files per record using a configurable split
method. Each segment inherits the parent record's labels.

Split methods (``--split-method``):
    two_halves   Two non-overlapping windows: 0–5 s, 5–10 s (default).
    overlap_2p5  Three windows with 2.5 s step: 0–5, 2.5–7.5, 5–10 s.
    overlap_1s   Sliding windows with 1 s step: 0–5, 1–6, …, 5–10 s.
    random       One random 5 s window per record (use ``--seed`` for reproducibility).

Outputs:
    waveforms/*.mat       fairseq-compatible mats with feats (12, 2500).
    metadata/samples.csv  Segment index, ecg_id, paths, and fold metadata.
    metadata/split_config.yaml  Split method and parameters used for this run.
    labels/labels.csv     Segment-level multi-hot labels.
    labels/y.npy          Float32 array aligned with labels.csv rows.
    labels/label_def.csv  Label names with train counts and pos_weight.
    labels/pos_weight.txt Train-set neg/pos ratio per label.
    labels/lead_mean.txt  Per-lead train-set mean for optional z-score norm.
    labels/lead_std.txt   Per-lead train-set std for optional z-score norm.
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import load_paths  # noqa: E402

SAMPLE_RATE = 500
SEGMENT_SEC = 5
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SEC
RECORD_SAMPLES = SEGMENT_SAMPLES * 2
N_LEADS = 12
TRAIN_FOLDS = set(range(1, 9))

SPLIT_METHODS = ("two_halves", "overlap_2p5", "overlap_1s", "random")
OVERLAP_2P5_STEP_SAMPLES = int(2.5 * SAMPLE_RATE)
OVERLAP_1S_STEP_SAMPLES = SAMPLE_RATE
MAX_SEGMENT_START = RECORD_SAMPLES - SEGMENT_SAMPLES


def segment_start_samples(
    method: str,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Return sample offsets (inclusive start) for 5 s windows within a 10 s record."""
    if method == "two_halves":
        return [0, SEGMENT_SAMPLES]
    if method == "overlap_2p5":
        return [0, OVERLAP_2P5_STEP_SAMPLES, SEGMENT_SAMPLES]
    if method == "overlap_1s":
        return list(range(0, MAX_SEGMENT_START + 1, OVERLAP_1S_STEP_SAMPLES))
    if method == "random":
        if rng is None:
            raise ValueError("random split method requires a NumPy RNG")
        return [int(rng.integers(0, MAX_SEGMENT_START + 1))]
    raise ValueError(f"Unknown split method: {method!r} (choices: {SPLIT_METHODS})")


def extract_segments(signal: np.ndarray, start_samples: list[int]) -> list[np.ndarray]:
    """Slice a (leads, samples) record into fixed-length segment arrays."""
    segments: list[np.ndarray] = []
    for start in start_samples:
        end = start + SEGMENT_SAMPLES
        if end > signal.shape[1]:
            raise ValueError(
                f"Segment start={start} exceeds record length {signal.shape[1]}"
            )
        segments.append(signal[:, start:end])
    return segments


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


def write_lead_normalization_stats(
    labels_dir: Path,
    train_sum: np.ndarray,
    train_sq_sum: np.ndarray,
    train_count: int,
) -> None:
    """Write per-lead train-set mean/std for fairseq z-score normalization."""
    if train_count == 0:
        raise ValueError("No train-fold segments found for lead statistics.")

    mean = train_sum / train_count
    var = np.maximum(train_sq_sum / train_count - mean**2, 0.0)
    std = np.sqrt(var)
    std = np.where(std > 0, std, 1.0)

    mean_path = labels_dir / "lead_mean.txt"
    std_path = labels_dir / "lead_std.txt"
    np.savetxt(mean_path, mean, fmt="%.8f")
    np.savetxt(std_path, std, fmt="%.8f")
    print("Saved lead normalization stats to:", mean_path, "and", std_path)


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
    labels_df = labels_df[
        ["idx", "ecg_id", "segment_idx", "start_sample", *label_cols]
    ]
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
        "--split-method",
        choices=SPLIT_METHODS,
        default="two_halves",
        help=(
            "How to extract 5 s segments from each 10 s record. "
            "All segments from the same record share its labels."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --split-method random (ignored otherwise).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records (for debugging).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing dataset even if split_method differs.",
    )
    return parser.parse_args()


def existing_split_method(metadata_dir: Path) -> str | None:
    """Return split_method from prior script-02 run, if any."""
    split_config_path = metadata_dir / "split_config.yaml"
    if split_config_path.is_file():
        with split_config_path.open() as f:
            config = yaml.safe_load(f) or {}
        method = config.get("split_method")
        return str(method) if method else None

    samples_path = metadata_dir / "samples.csv"
    if samples_path.is_file():
        samples = pd.read_csv(samples_path, usecols=["split_method"])
        if not samples.empty:
            return str(samples.iloc[0]["split_method"])
    return None


def assert_safe_to_write(
    metadata_dir: Path,
    split_method: str,
    processed_root: Path,
    *,
    force: bool,
) -> None:
    """Refuse to clobber a dataset prepared with a different split method."""
    prior_method = existing_split_method(metadata_dir)
    if prior_method is None or prior_method == split_method:
        return

    message = (
        f"Refusing to write split_method={split_method!r} into {processed_root}\n"
        f"  Existing data was prepared with split_method={prior_method!r}.\n"
        f"  Segments share filenames (e.g. 00009_seg1.mat) but different windows,\n"
        f"  so reusing this folder would corrupt the prior dataset.\n"
        f"  Fix: set a separate processed_root in configs/paths.yaml, e.g.\n"
        f"    processed_root: .../ptbxl_subclass_{{split_method}}\n"
        f"  Then re-run. Pass --force to overwrite anyway."
    )
    if force:
        print(f"Warning: {message}\n")
        return
    raise SystemExit(message)


def main() -> int:
    args = parse_args()
    paths = load_paths()
    configured_split = paths.get("split_method")
    if configured_split is not None and configured_split != args.split_method:
        print(
            f"Warning: --split-method {args.split_method!r} differs from "
            f"paths.yaml split_method {configured_split!r}. "
            "Update configs/paths.yaml processed_root before training."
        )

    raw_ptbxl = paths["raw_ptbxl"]
    labels_dir = paths["labels_dir"]
    metadata_dir = paths["metadata_dir"]
    waveform_dir = paths["waveform_dir"]
    processed_root = paths["processed_root"]

    assert_safe_to_write(
        metadata_dir,
        args.split_method,
        processed_root,
        force=args.force,
    )

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

    rng = np.random.default_rng(args.seed) if args.split_method == "random" else None
    if args.split_method == "random":
        segments_per_record = 1
        segment_starts_sec: list[float] | str = (
            f"uniform random offset in [0, {MAX_SEGMENT_START / SAMPLE_RATE:.1f}] s"
        )
    else:
        template_starts = segment_start_samples(args.split_method)
        segments_per_record = len(template_starts)
        segment_starts_sec = [start / SAMPLE_RATE for start in template_starts]

    split_config = {
        "split_method": args.split_method,
        "segment_sec": SEGMENT_SEC,
        "sample_rate": SAMPLE_RATE,
        "segment_samples": SEGMENT_SAMPLES,
        "record_samples": RECORD_SAMPLES,
        "segments_per_record": segments_per_record,
        "segment_starts_sec": segment_starts_sec,
        "seed": args.seed if args.split_method == "random" else None,
    }
    split_config_path = metadata_dir / "split_config.yaml"
    with split_config_path.open("w") as f:
        yaml.safe_dump(split_config, f, sort_keys=False)
    print("Split method:", args.split_method)
    print("Segment starts (s):", split_config["segment_starts_sec"])

    sample_rows: list[dict] = []
    label_rows: list[dict] = []
    skipped = 0
    segment_idx_global = 0
    train_sum = np.zeros(N_LEADS, dtype=np.float64)
    train_sq_sum = np.zeros(N_LEADS, dtype=np.float64)
    train_count = 0

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

        start_samples = segment_start_samples(args.split_method, rng=rng)
        segments = extract_segments(signal, start_samples)

        for segment_idx, (start, feats) in enumerate(zip(start_samples, segments)):
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
                "start_sample": start,
                "start_sec": start / SAMPLE_RATE,
                "split_method": args.split_method,
                "patient_id": int(record_meta.patient_id),
                "strat_fold": int(record_meta.strat_fold),
                "filename_hr": record_meta.filename_hr,
                "mat_path": str(mat_path.relative_to(paths["processed_root"])),
            })

            label_row = {
                "idx": segment_idx_global,
                "ecg_id": ecg_id,
                "segment_idx": segment_idx,
                "start_sample": start,
                "strat_fold": int(record_meta.strat_fold),
            }
            for label in label_cols:
                label_row[label] = int(row[label])
            label_rows.append(label_row)

            if int(record_meta.strat_fold) in TRAIN_FOLDS:
                train_sum += feats.sum(axis=1)
                train_sq_sum += np.square(feats, dtype=np.float64).sum(axis=1)
                train_count += feats.shape[1]

            segment_idx_global += 1

        if record_idx % 1000 == 0:
            print(f"Processed {record_idx}/{len(record_labels)} records...")

    if not sample_rows:
        print("No segments were created.")
        return 1

    samples_df = pd.DataFrame(sample_rows)
    samples_df.to_csv(metadata_dir / "samples.csv", index=False)

    write_segment_labels(labels_dir, label_rows, label_cols)
    write_lead_normalization_stats(labels_dir, train_sum, train_sq_sum, train_count)

    print("Saved waveforms to:", waveform_dir)
    print("Saved samples to:", metadata_dir / "samples.csv")
    print("Saved split config to:", split_config_path)
    print("num segments:", len(sample_rows))
    if skipped:
        print("skipped records:", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
