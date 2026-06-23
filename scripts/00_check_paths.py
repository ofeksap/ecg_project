#!/usr/bin/env python3
"""Verify that required PTB-XL and ECG-FM paths exist before running the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml"


def load_paths() -> dict[str, Path]:
    with PATHS_FILE.open() as f:
        raw = yaml.safe_load(f)
    return {key: Path(value) for key, value in raw.items()}


def check_path(label: str, path: Path, *, expect_dir: bool = False) -> bool:
    if expect_dir:
        ok = path.is_dir()
    else:
        ok = path.is_file()
    status = "OK" if ok else "MISSING"
    print(f"[{status}] {label}: {path}")
    return ok


def main() -> int:
    if not PATHS_FILE.is_file():
        print(f"[MISSING] paths config: {PATHS_FILE}")
        return 1

    paths = load_paths()
    raw_ptbxl = paths["raw_ptbxl"]
    pretrained_model = paths["pretrained_model"]
    mimic_finetuned_model = paths.get("mimic_finetuned_model")

    checks = [
        ("PTB-XL raw folder", raw_ptbxl, True),
        ("ptbxl_database.csv", raw_ptbxl / "ptbxl_database.csv", False),
        ("scp_statements.csv", raw_ptbxl / "scp_statements.csv", False),
        ("records500/", raw_ptbxl / "records500", True),
        ("ECG-FM pretrained checkpoint", pretrained_model, False),
    ]

    results = [check_path(label, path, expect_dir=expect_dir) for label, path, expect_dir in checks]

    if mimic_finetuned_model is not None:
        mimic_ok = check_path("ECG-FM MIMIC-finetuned checkpoint", mimic_finetuned_model, expect_dir=False)
        if not mimic_ok:
            print("[WARN] mimic_iv_ecg_finetuned.pt is required for script 06 only.")
        results.append(True)
    if all(results):
        print("\nAll required paths exist.")
        return 0

    print("\nOne or more required paths are missing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
