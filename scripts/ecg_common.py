"""Shared helpers for PTB-XL ECG-FM training, inference, and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml"

SPLIT_FOLDS = {
    "train": set(range(1, 9)),
    "valid": {9},
    "test": {10},
}


NON_PATH_KEYS = frozenset({"label_mode", "split_method"})


def resolve_path_templates(raw: dict[str, object]) -> dict[str, str]:
    """Expand ``{key}`` and ``<key>`` placeholders in ``paths.yaml`` string values."""
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            resolved[key] = ""
        else:
            resolved[key] = str(value)

    for _ in range(len(resolved) + 1):
        changed = False
        for key, value in list(resolved.items()):
            new_value = value
            for ref_key, ref_value in resolved.items():
                for open_brace, close_brace in (("{", "}"), ("<", ">")):
                    token = f"{open_brace}{ref_key}{close_brace}"
                    if token in new_value:
                        new_value = new_value.replace(token, ref_value)
            if new_value != value:
                resolved[key] = new_value
                changed = True
        if not changed:
            break

    for key, value in resolved.items():
        if key in NON_PATH_KEYS:
            continue
        if "{" in value or "<" in value:
            raise ValueError(
                f"Unresolved placeholder in paths.yaml key {key!r}: {value!r}"
            )
    return resolved


def load_paths(paths_file: Path | None = None) -> dict[str, Path | str]:
    """Load project paths from ``configs/paths.yaml`` with placeholder expansion."""
    config_path = paths_file or PATHS_FILE
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    resolved = resolve_path_templates(raw)
    return {
        key: value if key in NON_PATH_KEYS else Path(value)
        for key, value in resolved.items()
    }


def assign_split(strat_fold: int) -> str:
    """Map a PTB-XL stratified fold to train, valid, or test."""
    for split_name, folds in SPLIT_FOLDS.items():
        if strat_fold in folds:
            return split_name
    raise ValueError(f"Unexpected strat_fold: {strat_fold}")


def load_samples(metadata_dir: Path, segment_length: int = 2500) -> pd.DataFrame:
    """Load and normalize ``metadata/samples.csv``."""
    samples = pd.read_csv(metadata_dir / "samples.csv")
    if "split" not in samples.columns:
        samples["split"] = samples["strat_fold"].map(assign_split)
    if "relpath" not in samples.columns:
        mat_names = samples["mat_path"].map(lambda path: Path(path).name)
        samples["relpath"] = samples["split"] + "/" + mat_names
    if "length" not in samples.columns:
        samples["length"] = segment_length
    return samples


def get_split_indices(samples: pd.DataFrame, split: str) -> np.ndarray:
    """Return row indices in ``samples`` belonging to a split."""
    return samples.index[samples["split"] == split].to_numpy()


def load_label_names(labels_dir: Path) -> list[str]:
    """Return ordered label names from ``label_def.csv``."""
    label_def = pd.read_csv(labels_dir / "label_def.csv")
    return label_def["name"].astype(str).tolist()


def load_test_ground_truth(
    labels_dir: Path,
    metadata_dir: Path,
    *,
    record_level: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Return test labels, test sample metadata, and label names.

    When ``record_level=True``, one row per 10 s ``ecg_id`` (segment labels must match).
    """
    y_test, test_samples, label_names = _load_test_segment_ground_truth(
        labels_dir, metadata_dir
    )
    if not record_level:
        return y_test, test_samples, label_names

    y_record, record_meta = aggregate_labels_by_record(y_test, test_samples)
    return y_record, record_meta, label_names


def _load_test_segment_ground_truth(
    labels_dir: Path,
    metadata_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    samples = load_samples(metadata_dir)
    test_samples = samples[samples["split"] == "test"].sort_values("idx").reset_index(drop=True)
    if test_samples.empty:
        raise ValueError("No test samples found in samples.csv")

    y = np.load(labels_dir / "y.npy")
    label_names = load_label_names(labels_dir)
    test_idx = test_samples["idx"].to_numpy()
    y_test = y[test_idx]
    return y_test, test_samples, label_names


def aggregate_labels_by_record(
    y_true: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """One label row per ``ecg_id``; segment labels for the same record must agree."""
    if y_true.shape[0] != len(meta):
        raise ValueError("y_true and meta must have the same number of rows")

    agg_labels: list[np.ndarray] = []
    record_rows: list[dict[str, int]] = []

    for ecg_id, group in meta.groupby("ecg_id", sort=True):
        rows = group.index.to_numpy()
        seg_labels = y_true[rows]
        if not np.all(seg_labels == seg_labels[0]):
            raise ValueError(f"Inconsistent labels across segments for ecg_id={ecg_id}")

        agg_labels.append(seg_labels[0])
        record_rows.append({
            "ecg_id": int(ecg_id),
            "idx": int(group["idx"].iloc[0]),
            "num_segments": int(len(group)),
        })

    return np.stack(agg_labels, axis=0).astype(np.float32), pd.DataFrame(record_rows)


def aggregate_logits_by_record(
    logits: np.ndarray,
    meta: pd.DataFrame,
    *,
    aggregate: str = "mean",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean-aggregate segment logits to one row per ``ecg_id``."""
    if aggregate != "mean":
        raise ValueError(f"Unsupported aggregation: {aggregate!r} (supported: 'mean')")
    if logits.shape[0] != len(meta):
        raise ValueError("logits and meta must have the same number of rows")

    agg_logits: list[np.ndarray] = []
    record_rows: list[dict[str, int]] = []

    for ecg_id, group in meta.groupby("ecg_id", sort=True):
        rows = group.index.to_numpy()
        agg_logits.append(logits[rows].mean(axis=0))
        record_rows.append({
            "ecg_id": int(ecg_id),
            "idx": int(group["idx"].iloc[0]),
            "num_segments": int(len(group)),
        })

    return np.stack(agg_logits, axis=0).astype(np.float32), pd.DataFrame(record_rows)


def save_test_predictions(
    out_dir: Path,
    logits: np.ndarray,
    test_meta: pd.DataFrame,
    label_names: list[str],
    threshold: float = 0.5,
    *,
    prefix: str = "",
) -> None:
    """Write standardized test prediction artifacts.

    Use ``prefix='record_'`` for 10 s record-level outputs aggregated from segments.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int8)

    np.save(out_dir / f"{prefix}test_logits.npy", logits.astype(np.float32))
    np.save(out_dir / f"{prefix}test_predictions.npy", preds)

    meta_cols = [col for col in ("idx", "ecg_id", "segment_idx", "num_segments") if col in test_meta.columns]
    pred_df = test_meta[meta_cols].copy()
    for i, label in enumerate(label_names):
        pred_df[label] = preds[:, i]
    pred_df.to_csv(out_dir / f"{prefix}test_predictions.csv", index=False)


def _ensure_fairseq_import(fairseq_signals_root: Path) -> None:
    root = str(fairseq_signals_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def run_fairseq_test_inference(
    checkpoint: Path,
    manifest_dir: Path,
    fairseq_signals_root: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "cuda",
) -> np.ndarray:
    """Run inference on the test manifest and return logits in manifest order."""
    _ensure_fairseq_import(fairseq_signals_root)

    import torch
    from fairseq_signals.utils import checkpoint_utils, utils

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and not use_cuda:
        raise RuntimeError("CUDA requested but not available.")

    overrides = {
        "task": {"data": str(manifest_dir)},
        "model_path": None,
        "no_pretrained_weights": True,
    }

    model, saved_cfg, task = checkpoint_utils.load_model_and_task(
        str(checkpoint),
        checkpoint_overrides=overrides,
        strict=False,
    )
    model.eval()
    if use_cuda:
        model.cuda()

    task.load_dataset(
        "test",
        combine=False,
        epoch=1,
        task_cfg=saved_cfg.task,
        label=False,
        shuffle=False,
    )
    dataset = task.dataset("test")

    batch_iterator = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=saved_cfg.dataset.max_tokens,
        max_signals=batch_size,
        ignore_invalid_inputs=saved_cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=saved_cfg.dataset.required_batch_size_multiple,
        seed=saved_cfg.common.seed,
        num_shards=1,
        shard_id=0,
        num_workers=num_workers,
        data_buffer_size=saved_cfg.dataset.data_buffer_size,
    )
    itr = batch_iterator.next_epoch_itr(shuffle=False)

    logits_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for sample in itr:
            if sample is None or len(sample) == 0:
                continue
            if use_cuda:
                sample = utils.move_to_cuda(sample)
            net_output = model(**sample["net_input"])
            logits = model.get_logits(net_output).float().cpu().numpy()
            logits_chunks.append(logits)

    if not logits_chunks:
        raise RuntimeError("No logits produced during test inference.")

    return np.concatenate(logits_chunks, axis=0)


def verify_test_logits(
    logits: np.ndarray,
    test_meta: pd.DataFrame,
    label_names: list[str],
) -> None:
    """Validate inference output shape."""
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {logits.shape}")
    if logits.shape[0] != len(test_meta):
        raise ValueError(
            f"logits rows ({logits.shape[0]}) != test samples ({len(test_meta)})"
        )
    if logits.shape[1] != len(label_names):
        raise ValueError(
            f"logits cols ({logits.shape[1]}) != num labels ({len(label_names)})"
        )


def build_fairseq_train_cmd(
    fairseq_signals_root: Path,
    manifest_dir: Path,
    label_dir: Path,
    output_dir: Path,
    num_labels: int,
    pos_weight: str,
    *,
    model_path: str | None,
    no_pretrained_weights: bool = False,
    normalize: bool = False,
    mean_path: Path | None = None,
    std_path: Path | None = None,
) -> list[str]:
    """Build a fairseq-hydra-train command list."""
    import shutil

    if normalize and (mean_path is None or std_path is None):
        raise ValueError("normalize=True requires mean_path and std_path")
    if normalize:
        if not mean_path.is_file() or not std_path.is_file():
            raise FileNotFoundError(
                f"Missing lead normalization stats: {mean_path} or {std_path}. "
                "Re-run scripts/02_prepare_ptbxl_waveforms.py."
            )

    train_bin = shutil.which("fairseq-hydra-train")
    if train_bin is None:
        candidate = fairseq_signals_root / "fairseq-hydra-train"
        if candidate.exists():
            train_bin = str(candidate)
        else:
            train_bin = "fairseq-hydra-train"

    config_dir = (
        fairseq_signals_root / "examples/w2v_cmsc/config/finetuning/ecg_transformer"
    )

    cmd = [
        train_bin,
        f"task.data={manifest_dir}",
        f"model.num_labels={num_labels}",
        "optimization.lr=[1e-06]",
        "optimization.max_epoch=140",
        "dataset.batch_size=16",
        "dataset.num_workers=5",
        "dataset.valid_subset=valid",
        "dataset.disable_validation=false",
        "distributed_training.distributed_world_size=1",
        "distributed_training.find_unused_parameters=True",
        f"checkpoint.save_dir={output_dir}",
        "checkpoint.save_interval=1",
        "checkpoint.keep_last_epochs=1",
        "checkpoint.best_checkpoint_metric=auroc",
        "checkpoint.maximize_best_checkpoint_metric=true",
        "common.log_format=csv",
        f'+task.label_file={label_dir / "y.npy"}',
        f"+criterion.pos_weight={pos_weight}",
        f"--config-dir={config_dir}",
        "--config-name=diagnosis",
    ]

    if no_pretrained_weights:
        cmd[1:1] = ["model.no_pretrained_weights=true", "model.model_path=null"]
    elif model_path is not None:
        cmd.insert(1, f"model.model_path={model_path}")

    if normalize:
        cmd[1:1] = [
            "task.normalize=true",
            f"+task.mean_path={mean_path}",
            f"+task.std_path={std_path}",
        ]

    return cmd
