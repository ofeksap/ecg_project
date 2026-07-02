#!/usr/bin/env python3
"""Visualize example PTB-XL waveforms and 5 s segment splits for each method.

Loads a full 10 s 12-lead recording from raw PTB-XL WFDB and shows how each
``split_method`` (two_halves, overlap_2p5, overlap_1s, random) carves it into
5 s segments. Also annotates the PTB-XL train/valid/test fold assignment.

Outputs (under ``--output-dir`` by default):
    ecg_{ecg_id:05d}_12lead_full.png       Full 10 s 12-lead layout
    ecg_{ecg_id:05d}_split_overview.png    Lead II with segment windows per method
    ecg_{ecg_id:05d}_split_segments.png    Extracted segment waveforms per method
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import assign_split, load_paths  # noqa: E402

SPLIT_METHODS = ("two_halves", "overlap_2p5", "overlap_1s", "random")
LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
LEAD_II_IDX = 1
SAMPLE_RATE = 500
SEGMENT_SEC = 5
RECORD_SEC = 10
RANDOM_SEED = 42


def _load_prepare_module():
    """Import helpers from ``02_prepare_ptbxl_waveforms.py`` (non-standard module name)."""
    module_path = SCRIPT_DIR / "02_prepare_ptbxl_waveforms.py"
    spec = importlib.util.spec_from_file_location("prepare_ptbxl_waveforms", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_ecg_id(
    ecg_id: int | None,
    raw_ptbxl: Path,
    record_labels_path: Path | None,
) -> int:
    """Return ``ecg_id`` or pick the first labeled record from ``record_labels.csv``."""
    if ecg_id is not None:
        return ecg_id

    if record_labels_path is not None and record_labels_path.is_file():
        labels = pd.read_csv(record_labels_path)
        return int(labels.iloc[0]["ecg_id"])

    meta = pd.read_csv(raw_ptbxl / "ptbxl_database.csv", index_col="ecg_id")
    return int(meta.index[0])


def load_record(raw_ptbxl: Path, ecg_id: int, prepare) -> tuple[np.ndarray, pd.Series]:
    """Load a 10 s WFDB record and its PTB-XL metadata row."""
    meta = pd.read_csv(raw_ptbxl / "ptbxl_database.csv", index_col="ecg_id")
    if ecg_id not in meta.index:
        raise ValueError(f"ecg_id={ecg_id} not found in ptbxl_database.csv")

    record_meta = meta.loc[ecg_id]
    record_path = raw_ptbxl / record_meta.filename_hr
    signal, _fs = prepare.read_wfdb_record(record_path)
    return signal, record_meta


def segment_windows(method: str, prepare, *, seed: int = RANDOM_SEED) -> list[tuple[int, float, float]]:
    """Return (start_sample, start_sec, end_sec) for each segment under ``method``."""
    rng = np.random.default_rng(seed) if method == "random" else None
    starts = prepare.segment_start_samples(method, rng=rng)
    windows: list[tuple[int, float, float]] = []
    for start in starts:
        start_sec = start / SAMPLE_RATE
        end_sec = start_sec + SEGMENT_SEC
        windows.append((start, start_sec, end_sec))
    return windows


def plot_12lead_full(
    signal: np.ndarray,
    ecg_id: int,
    record_meta: pd.Series,
    out_path: Path,
) -> None:
    """Plot the full 10 s recording in a standard 3×4 12-lead layout."""
    n_samples = signal.shape[1]
    time = np.arange(n_samples) / SAMPLE_RATE
    ml_split = assign_split(int(record_meta.strat_fold))

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    fig.suptitle(
        f"PTB-XL ecg_id={ecg_id}  |  patient_id={int(record_meta.patient_id)}  "
        f"|  strat_fold={int(record_meta.strat_fold)} ({ml_split})  |  10 s @ 500 Hz",
        fontsize=13,
        y=0.98,
    )

    for lead_idx, ax in enumerate(axes.ravel()):
        ax.plot(time, signal[lead_idx], color="#1a1a1a", linewidth=0.7)
        ax.set_ylabel(LEAD_NAMES[lead_idx], rotation=0, labelpad=18, fontsize=10, va="center")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlim(0, RECORD_SEC)

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_overview(
    signal: np.ndarray,
    ecg_id: int,
    prepare,
    methods: tuple[str, ...],
    out_path: Path,
    *,
    lead_idx: int = LEAD_II_IDX,
    seed: int = RANDOM_SEED,
) -> None:
    """Plot Lead II with shaded segment windows, one row per split method."""
    n_samples = signal.shape[1]
    time = np.arange(n_samples) / SAMPLE_RATE
    lead_trace = signal[lead_idx]

    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 2.6 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")

    for ax, method in zip(axes, methods):
        windows = segment_windows(method, prepare, seed=seed)
        ax.plot(time, lead_trace, color="#1a1a1a", linewidth=0.9, zorder=1)
        legend_handles: list[mpatches.Patch] = []

        for seg_idx, (_start, start_sec, end_sec) in enumerate(windows):
            color = cmap(seg_idx % 10)
            ax.axvspan(start_sec, end_sec, alpha=0.25, color=color, zorder=0)
            mid = (start_sec + end_sec) / 2
            ax.text(
                mid,
                ax.get_ylim()[1] * 0.92,
                f"seg{seg_idx}\n{start_sec:.1f}–{end_sec:.1f}s",
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
            legend_handles.append(
                mpatches.Patch(color=color, alpha=0.5, label=f"seg{seg_idx}: {start_sec:.1f}–{end_sec:.1f} s")
            )

        ax.set_ylabel(f"Lead {LEAD_NAMES[lead_idx]}\n(mV)", fontsize=10)
        ax.set_title(f"{method}  ({len(windows)} segment{'s' if len(windows) != 1 else ''})", loc="left")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Segment window placement on ecg_id={ecg_id} — Lead {LEAD_NAMES[lead_idx]}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_segments(
    signal: np.ndarray,
    ecg_id: int,
    prepare,
    methods: tuple[str, ...],
    out_path: Path,
    *,
    lead_idx: int = LEAD_II_IDX,
    seed: int = RANDOM_SEED,
) -> None:
    """Grid of extracted segment waveforms: columns = methods, rows = segment index."""
    max_segments = max(len(segment_windows(m, prepare, seed=seed)) for m in methods)
    n_methods = len(methods)

    fig, axes = plt.subplots(
        max_segments,
        n_methods,
        figsize=(3.4 * n_methods, 2.0 * max_segments),
        sharex=True,
        squeeze=False,
    )

    cmap = plt.get_cmap("tab10")
    seg_time = np.arange(SAMPLE_RATE * SEGMENT_SEC) / SAMPLE_RATE

    for col, method in enumerate(methods):
        windows = segment_windows(method, prepare, seed=seed)
        for row in range(max_segments):
            ax = axes[row, col]
            if row >= len(windows):
                ax.axis("off")
                continue

            start, start_sec, end_sec = windows[row]
            seg = signal[lead_idx, start : start + SAMPLE_RATE * SEGMENT_SEC]
            color = cmap(row % 10)
            ax.plot(seg_time, seg, color=color, linewidth=0.8)
            ax.set_ylabel("mV", fontsize=8)
            ax.grid(True, alpha=0.25, linewidth=0.5)

            if row == 0:
                ax.set_title(method, fontsize=11, fontweight="bold")
            if col == 0:
                ax.text(
                    -0.28,
                    0.5,
                    f"seg{row}\n{start_sec:.1f}–{end_sec:.1f}s",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Time within segment (s)")

    fig.suptitle(
        f"Extracted 5 s segments — ecg_id={ecg_id}, Lead {LEAD_NAMES[lead_idx]}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PTB-XL waveforms and segment splits for each split method.",
    )
    parser.add_argument(
        "--ecg-id",
        type=int,
        default=None,
        help="PTB-XL ecg_id to plot (default: first record in record_labels.csv)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SPLIT_METHODS,
        default=list(SPLIT_METHODS),
        help="Split methods to visualize (default: all)",
    )
    parser.add_argument(
        "--lead",
        default="II",
        choices=LEAD_NAMES,
        help="Lead used for split overview and segment plots (default: II)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="RNG seed for the random split method (default: 42, matches preprocessing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs (default: <project_root>/figures/waveform_splits)",
    )
    parser.add_argument(
        "--paths-file",
        type=Path,
        default=None,
        help="Optional override for configs/paths.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths_file)
    prepare = _load_prepare_module()

    raw_ptbxl = paths["raw_ptbxl"]
    processed_root = Path(str(paths["processed_root"]))
    record_labels_path = processed_root / "labels" / "record_labels.csv"

    ecg_id = resolve_ecg_id(args.ecg_id, raw_ptbxl, record_labels_path)
    signal, record_meta = load_record(raw_ptbxl, ecg_id, prepare)

    output_dir = args.output_dir or (paths["project_root"] / "figures" / "waveform_splits")
    lead_idx = LEAD_NAMES.index(args.lead)
    methods = tuple(args.methods)

    prefix = f"ecg_{ecg_id:05d}"
    full_path = output_dir / f"{prefix}_12lead_full.png"
    overview_path = output_dir / f"{prefix}_split_overview.png"
    segments_path = output_dir / f"{prefix}_split_segments.png"

    ml_split = assign_split(int(record_meta.strat_fold))
    print(f"ecg_id={ecg_id}  patient_id={int(record_meta.patient_id)}  "
          f"strat_fold={int(record_meta.strat_fold)} ({ml_split})")
    for method in methods:
        windows = segment_windows(method, prepare, seed=args.seed)
        ranges = ", ".join(f"seg{i} [{s:.1f}–{e:.1f}s]" for i, (_, s, e) in enumerate(windows))
        print(f"  {method}: {len(windows)} segments — {ranges}")

    plot_12lead_full(signal, ecg_id, record_meta, full_path)
    plot_split_overview(signal, ecg_id, prepare, methods, overview_path, lead_idx=lead_idx, seed=args.seed)
    plot_split_segments(signal, ecg_id, prepare, methods, segments_path, lead_idx=lead_idx, seed=args.seed)

    print(f"\nSaved figures to {output_dir}/")
    print(f"  {full_path.name}")
    print(f"  {overview_path.name}")
    print(f"  {segments_path.name}")


if __name__ == "__main__":
    main()
