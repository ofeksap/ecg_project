#!/usr/bin/env python3
"""Convert ECG-FM MIMIC finetuned checkpoint for local PTB-XL fine-tuning.

The released ``mimic_iv_ecg_finetuned.pt`` is a full 17-label classifier and
stores a stale upstream ``model_path`` from the authors' cluster. Fairseq
expects an encoder-only pretraining checkpoint (like
``mimic_iv_ecg_physionet_pretrained.pt``) when starting a new classification
run.

This script writes ``mimic_iv_ecg_finetuned_encoder.pt``: encoder weights from
the MIMIC release wrapped in the same on-disk format as the PhysioNet
pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ecg_common import load_paths  # noqa: E402


def extract_encoder_state(
    classifier_state: dict[str, torch.Tensor],
    template_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    encoder_state: dict[str, torch.Tensor] = {}
    for key, value in classifier_state.items():
        if key.startswith("encoder.encoder."):
            encoder_state["encoder." + key[len("encoder.encoder.") :]] = value
        elif key.startswith("encoder."):
            rest = key[len("encoder.") :]
            if rest.startswith(
                ("feature_extractor.", "conv_pos.", "layer_norm.", "post_extract_proj.")
            ):
                encoder_state[rest] = value
    if not encoder_state:
        raise ValueError("No encoder weights found in classifier checkpoint.")

    # Wav2Vec2 CMSC checkpoints include mask_emb; fairseq loads it strictly,
    # then remove_pretraining_modules() drops it before fine-tuning.
    if "mask_emb" in template_state:
        encoder_state["mask_emb"] = template_state["mask_emb"]

    return encoder_state


def build_encoder_checkpoint(
    mimic_classifier_path: Path,
    physionet_pretrained_path: Path,
    output_path: Path,
) -> None:
    mimic = torch.load(mimic_classifier_path, map_location="cpu", weights_only=False)
    phys = torch.load(physionet_pretrained_path, map_location="cpu", weights_only=False)

    encoder_state = extract_encoder_state(mimic["model"], phys["model"])
    out = copy.deepcopy(phys)
    out["model"] = encoder_state
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)


def parse_args() -> argparse.Namespace:
    paths = load_paths()
    default_src = paths["project_root"] / "checkpoints/ecgfm/mimic_iv_ecg_finetuned.pt"
    default_phys = Path(paths["pretrained_model"])
    default_out = paths["project_root"] / "checkpoints/ecgfm/mimic_iv_ecg_finetuned_encoder.pt"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_src)
    parser.add_argument("--template", type=Path, default=default_phys)
    parser.add_argument("--output", type=Path, default=default_out)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        print(f"Missing source checkpoint: {args.source}")
        return 1
    if not args.template.is_file():
        print(f"Missing template checkpoint: {args.template}")
        return 1

    build_encoder_checkpoint(args.source, args.template, args.output)
    print(f"Wrote encoder checkpoint: {args.output}")
    print("Update mimic_finetuned_model in configs/paths.yaml to this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
