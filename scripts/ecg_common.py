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


def load_paths() -> dict[str, Path | str]:
    """Load project paths from ``configs/paths.yaml``."""
    with PATHS_FILE.open() as f:
        raw = yaml.safe_load(f)
    return {
        key: Path(value) if key != "label_mode" else value
        for key, value in raw.items()
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
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Return test labels, test sample metadata, and label names."""
    samples = load_samples(metadata_dir)
    test_samples = samples[samples["split"] == "test"].sort_values("idx").reset_index(drop=True)
    if test_samples.empty:
        raise ValueError("No test samples found in samples.csv")

    y = np.load(labels_dir / "y.npy")
    label_names = load_label_names(labels_dir)
    test_idx = test_samples["idx"].to_numpy()
    y_test = y[test_idx]
    return y_test, test_samples, label_names


def save_test_predictions(
    out_dir: Path,
    logits: np.ndarray,
    test_meta: pd.DataFrame,
    label_names: list[str],
    threshold: float = 0.5,
) -> None:
    """Write standardized test prediction artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int8)

    np.save(out_dir / "test_logits.npy", logits.astype(np.float32))
    np.save(out_dir / "test_predictions.npy", preds)

    pred_df = test_meta[["idx", "ecg_id", "segment_idx"]].copy()
    for i, label in enumerate(label_names):
        pred_df[label] = preds[:, i]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)


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
) -> list[str]:
    """Build a fairseq-hydra-train command list."""
    import shutil

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
        "checkpoint.keep_last_epochs=0",
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

    return cmd
