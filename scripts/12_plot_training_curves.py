#!/usr/bin/env python3
"""Plot train/valid loss and validation AUROC from fairseq-signals CSV logs.

Reads ``train.csv`` and ``valid.csv`` from a Hydra output directory under
``outputs/<date>/<time>/`` and saves a two-panel training curve PNG.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import load_paths  # noqa: E402


def find_latest_log_dir(outputs_root: Path) -> Path:
    """Return the newest Hydra log directory that contains ``train.csv``."""
    candidates = sorted(
        (p.parent for p in outputs_root.glob("*/*/train.csv") if p.is_file()),
        key=lambda path: path.relative_to(outputs_root).as_posix(),
    )
    if not candidates:
        raise FileNotFoundError(f"No train.csv logs found under {outputs_root}")
    return candidates[-1]


def infer_batches_per_epoch(train: pd.DataFrame) -> int:
    """Infer optimizer steps per epoch from fairseq ``train.csv`` step spacing."""
    steps = train["step"].astype(int).to_numpy()
    if steps.size == 0:
        raise ValueError("train.csv is empty")

    if steps.size == 1:
        return int(steps[0])

    diffs = [steps[i] - steps[i - 1] for i in range(1, len(steps))]
    if len(set(diffs[: min(3, len(diffs))])) == 1:
        return int(diffs[0])

    return int(steps[0])


def load_training_logs(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Load train/valid CSV logs and attach a 1-based ``epoch`` column."""
    train_path = log_dir / "train.csv"
    valid_path = log_dir / "valid.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing train log: {train_path}")
    if not valid_path.is_file():
        raise FileNotFoundError(f"Missing valid log: {valid_path}")

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    batches_per_epoch = infer_batches_per_epoch(train)

    train = train.copy()
    valid = valid.copy()
    train["epoch"] = train["step"] // batches_per_epoch
    valid["epoch"] = valid["step"] // batches_per_epoch
    return train, valid, batches_per_epoch


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w.-]+", "_", text.strip().lower())
    return slug.strip("_") or "training_curve"


def default_output_path(log_dir: Path, title: str | None, project_root: Path) -> Path:
    if title:
        stem = f"{_slugify(title)}_train_valid"
    else:
        rel = log_dir.relative_to(project_root / "outputs")
        stem = f"{rel.as_posix().replace('/', '_')}_train_valid"
    return project_root / "results" / "figures" / "training_curves" / f"{stem}.png"


def plot_training_curves(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    title: str,
    metric: str,
    output_path: Path,
    dpi: int,
) -> dict[str, float | int]:
    if metric not in valid.columns:
        raise ValueError(
            f"Metric '{metric}' not found in valid.csv columns: {list(valid.columns)}"
        )

    best_idx = valid[metric].idxmax()
    best_ep = int(valid.loc[best_idx, "epoch"])
    best_value = float(valid.loc[best_idx, metric])
    latest_ep = int(valid["epoch"].max())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(train["epoch"], train["loss"], label="train", color="#2563eb", linewidth=1.5)
    axes[0].plot(valid["epoch"], valid["loss"], label="valid", color="#dc2626", linewidth=1.5)
    axes[0].axvline(
        best_ep,
        color="#16a34a",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label=f"best {metric} (ep {best_ep})",
    )
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — train vs valid loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(valid["epoch"], valid[metric], label=f"valid {metric}", color="#7c3aed", linewidth=1.5)
    if f"best_{metric}" in valid.columns:
        axes[1].plot(
            valid["epoch"],
            valid[f"best_{metric}"],
            label=f"best {metric} so far",
            color="#16a34a",
            linewidth=1,
            linestyle="--",
            alpha=0.8,
        )
    axes[1].axvline(best_ep, color="#16a34a", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric.upper())
    axes[1].set_title(f"Validation {metric.upper()} (checkpoint selection metric)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    metric_values = valid[metric].dropna()
    if not metric_values.empty:
        ymin = max(0.0, float(metric_values.min()) - 0.02)
        ymax = min(1.0, float(metric_values.max()) + 0.01)
        if ymax > ymin:
            axes[1].set_ylim(ymin, ymax)

    fig.text(
        0.02,
        0.02,
        (
            f"Best: epoch {best_ep}, {metric}={best_value:.4f}  |  "
            f"Latest: epoch {latest_ep}, {metric}={valid.iloc[-1][metric]:.4f}, "
            f"valid loss={valid.iloc[-1]['loss']:.1f}"
        ),
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "best_epoch": best_ep,
        "best_metric": best_value,
        "latest_epoch": latest_ep,
        "latest_metric": float(valid.iloc[-1][metric]),
        "latest_valid_loss": float(valid.iloc[-1]["loss"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/valid loss and validation AUROC from fairseq CSV logs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Hydra output directory containing train.csv and valid.csv",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest outputs/<date>/<time>/ directory with train.csv",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title and default output filename stem (e.g. overlap_1s / pretrained_exp_001)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: results/figures/training_curves/<title>_train_valid.png)",
    )
    parser.add_argument(
        "--metric",
        default="auroc",
        help="Validation metric used to mark the best epoch (default: auroc)",
    )
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help="Override auto-detected optimizer steps per epoch",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution (default: 150)",
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
    if args.log_dir is None and not args.latest:
        raise SystemExit("Provide --log-dir or use --latest.")

    paths = load_paths(args.paths_file)
    project_root = Path(paths["project_root"])
    outputs_root = project_root / "outputs"

    log_dir = args.log_dir or find_latest_log_dir(outputs_root)
    train, valid, inferred_batches = load_training_logs(log_dir)
    if args.batches_per_epoch is not None:
        batches_per_epoch = args.batches_per_epoch
        train["epoch"] = train["step"] // batches_per_epoch
        valid["epoch"] = valid["step"] // batches_per_epoch

    title = args.title or log_dir.relative_to(outputs_root).as_posix()
    output_path = args.output or default_output_path(log_dir, args.title, project_root)

    summary = plot_training_curves(
        train,
        valid,
        title=title,
        metric=args.metric,
        output_path=output_path,
        dpi=args.dpi,
    )

    print(f"Log dir: {log_dir}")
    print(f"Batches per epoch: {args.batches_per_epoch or inferred_batches}")
    print(f"Saved: {output_path}")
    print(
        "Best epoch {best_epoch}: {metric}={best_metric:.4f} | "
        "Latest epoch {latest_epoch}: {metric}={latest_metric:.4f}, "
        "valid loss={latest_valid_loss:.1f}".format(metric=args.metric, **summary)
    )


if __name__ == "__main__":
    main()
