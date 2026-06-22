"""Build record-level multi-label targets from PTB-XL SCP codes.

Reads ``ptbxl_database.csv`` and maps each record's SCP codes to diagnostic
labels using ``scp_statements.csv``. Uses ``label_mode`` from
``configs/paths.yaml`` (``diagnostic_subclass`` for this project). Labels with
fewer than ``MIN_TRAIN_COUNT`` positive examples in training folds
(strat_fold 1-8) are dropped to avoid extremely rare classes.

Outputs (written to ``labels_dir`` from ``configs/paths.yaml``):
    record_labels.csv  Per-record index, ecg_id, and multi-hot label columns.

Segment-level ``labels.csv``, ``y.npy``, ``label_def.csv``, and
``pos_weight.txt`` are produced by ``02_prepare_ptbxl_waveforms.py`` after
splitting each 10 s record into two 5 s segments.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml"
MIN_TRAIN_COUNT = 50


def load_paths() -> dict[str, Path | str]:
    with PATHS_FILE.open() as f:
        raw = yaml.safe_load(f)
    return {
        key: Path(value) if key != "label_mode" else value
        for key, value in raw.items()
    }


def extract_labels(scp_dict: dict, diag_scp: pd.DataFrame, label_mode: str) -> list[str]:
    """Map an SCP-code dictionary to sorted diagnostic labels for one record."""
    labels = set()

    for code in scp_dict.keys():
        if code not in diag_scp.index:
            continue

        if label_mode == "diagnostic_superclass":
            label = diag_scp.loc[code, "diagnostic_class"]
        elif label_mode == "diagnostic_subclass":
            label = diag_scp.loc[code, "diagnostic_subclass"]
        elif label_mode == "scp_code":
            label = code
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        if pd.notna(label):
            labels.add(str(label))

    return sorted(labels)


def main() -> int:
    paths = load_paths()
    raw_ptbxl = paths["raw_ptbxl"]
    out_dir = paths["labels_dir"]
    label_mode = paths["label_mode"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_ptbxl / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(raw_ptbxl / "scp_statements.csv", index_col=0)
    diag_scp = scp[scp["diagnostic"] == 1]

    df["labels"] = df["scp_codes"].apply(
        lambda scp_dict: extract_labels(scp_dict, diag_scp, label_mode)
    )

    all_labels = sorted({label for labels in df["labels"] for label in labels})
    for label in all_labels:
        df[label] = df["labels"].apply(lambda xs, lbl=label: int(lbl in xs))

    train_mask = df["strat_fold"].isin(range(1, 9))
    train_counts = df.loc[train_mask, all_labels].sum(axis=0)
    selected_labels = train_counts[train_counts >= MIN_TRAIN_COUNT].index.tolist()

    print(f"label_mode: {label_mode}")
    print("Selected labels:")
    for label in selected_labels:
        print(f"{label}: {int(train_counts[label])} train positives")

    labels_df = df[selected_labels].copy()
    labels_df.insert(0, "ecg_id", labels_df.index)
    labels_df.insert(0, "idx", np.arange(len(labels_df)))
    labels_df.to_csv(out_dir / "record_labels.csv", index=False)

    print("Saved record labels to:", out_dir / "record_labels.csv")
    print("num records:", len(labels_df))
    print("num labels:", len(selected_labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
