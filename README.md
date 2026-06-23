ECG fine-tuning project

Tal Noy, Ofek Sapir

Dataset:
- PTB-XL

Baselines:
- ECG-FM fine-tuned model over MIMIC-IV-ECG with classification head adjusted to PTB-XL labels
- Transformer initliazied randomly

Compute: 
- GPU Model: NVIDIA GeForce RTX 3080 Ti with 12 GB total VRAM
- Driver & CUDA: Nvidia driver 535.274.02, supports CUDA 12.2

---

## Overview

Fine-tune [ECG-FM](https://github.com/bowang-lab/ECG-FM) on PTB-XL **diagnostic subclass** labels using [fairseq-signals](https://github.com/Jwoo5/fairseq-signals). Each 10 s ECG record is split into two **5 s segments at 500 Hz** (2500 samples × 12 leads). Splits follow PTB-XL `strat_fold`: train 1–8, valid 9, test 10.

Two ECG-FM starting points are compared:

| Experiment | Starting checkpoint | Script |
|------------|--------------------|--------|
| PhysioNet-pretrained | `mimic_iv_ecg_physionet_pretrained.pt` | `05_finetune_ecgfm_pretrained.sh` |
| MIMIC-finetuned | `mimic_iv_ecg_finetuned_encoder.pt` (converted locally) | `06_finetune_ecgfm_mimic_finetuned.sh` |

A randomly initialized `ecg_transformer_classifier` baseline uses the same data and fairseq config (`08_train_transformer_baseline.py`).

## Repository layout

```
configs/paths.yaml          # paths for data, checkpoints, and run outputs
scripts/                    # data prep, training, inference, evaluation
external/fairseq-signals/   # training / inference engine (install with pip install -e .)
external/ECG-FM/            # paper repo + HuggingFace checkpoint references
checkpoints/ecgfm/          # downloaded ECG-FM weights (gitignored)
data/processed/             # labels, waveforms, manifests (gitignored)
runs/                       # training checkpoints and predictions (gitignored)
outputs/                    # Hydra run logs from fairseq-hydra-train (gitignored)
```

Edit `configs/paths.yaml` if your project root or data locations differ from the defaults.

## Setup

1. **PTB-XL** — download v1.0.3 under `data/raw/physionet.org/files/ptb-xl/1.0.3/`.
2. **fairseq-signals** — clone into `external/fairseq-signals` and install:
   ```bash
   cd external/fairseq-signals && pip install -e .
   ```
3. **ECG-FM checkpoints** — place in `checkpoints/ecgfm/`:
   - `mimic_iv_ecg_physionet_pretrained.pt`
   - `mimic_iv_ecg_finetuned.pt` (original release; script 06 converts it automatically)
4. **Environment** — fairseq needs `fairseq-signals` on `PYTHONPATH` (training scripts set this). Example:
   ```bash
   export PYTHONPATH=/media/2TB/ecg_project/external/fairseq-signals:$PYTHONPATH
   ```

Verify paths:

```bash
python3 scripts/00_check_paths.py
```

## Pipeline

Run from the project root, in order:

| Step | Script | Purpose |
|------|--------|---------|
| 00 | `00_check_paths.py` | Sanity-check raw data and checkpoints |
| 01 | `01_make_ptbxl_labels.py` | Record-level labels from SCP codes (`diagnostic_subclass`) |
| 02 | `02_prepare_ptbxl_waveforms.py` | Split waveforms → 5 s `.mat` segments, `y.npy`, `pos_weight.txt` |
| 03 | `03_make_manifests.py` | fairseq `train/valid/test.tsv` + symlink split dirs under `waveforms/` |
| 04 | `04_check_dataset.py` | Validate alignment of labels, manifests, and mats |
| 05 | `05_finetune_ecgfm_pretrained.sh` | Fine-tune from PhysioNet-pretrained ECG-FM |
| 06 | `06_finetune_ecgfm_mimic_finetuned.sh` | Fine-tune from MIMIC-finetuned ECG-FM |
| 08 | `08_train_transformer_baseline.py` | Random-init transformer baseline |
| 09 | `09_predict_ecgfm.py` | Test-set inference → `test_logits.npy`, `test_predictions.csv` |
| 010 | `010_evaluate_predictions.py` | AUROC / AUPRC / F1 metrics from saved predictions |

Script `06_patch_mimic_checkpoint.py` converts the released MIMIC classifier checkpoint into an encoder-only file (`mimic_iv_ecg_finetuned_encoder.pt`) for local fine-tuning. Script 06 runs this automatically when needed.

### Data prep (full dataset)

```bash
python3 scripts/01_make_ptbxl_labels.py
python3 scripts/02_prepare_ptbxl_waveforms.py
python3 scripts/03_make_manifests.py
python3 scripts/04_check_dataset.py
```

### Training

```bash
bash scripts/05_finetune_ecgfm_pretrained.sh 2>&1 | tee runs/ecgfm_ptbxl_subclass/pretrained_exp_001/train.log

bash scripts/06_finetune_ecgfm_mimic_finetuned.sh 2>&1 | tee runs/ecgfm_ptbxl_subclass/mimic_finetuned_exp_001/train.log

python3 scripts/08_train_transformer_baseline.py 2>&1 | tee runs/ecgfm_ptbxl_subclass/transformer_baseline_exp_001/train.log
```

Checkpoints are written to the `output_dir_*` paths in `configs/paths.yaml`. Best model is selected by validation **AUROC** (`checkpoint_best.pt`).

### Inference and evaluation

```bash
python3 scripts/09_predict_ecgfm.py \
  --checkpoint runs/ecgfm_ptbxl_subclass/pretrained_exp_001/checkpoint_best.pt

python3 scripts/010_evaluate_predictions.py \
  --predictions-dir runs/ecgfm_ptbxl_subclass/pretrained_exp_001/predictions
```

For the MIMIC experiment, point `--checkpoint` at `mimic_finetuned_exp_001/checkpoint_best.pt`.

## Outputs

| Location | Contents |
|----------|----------|
| `runs/<experiment>/` | `checkpoint_best.pt`, per-epoch checkpoints, `train.log` |
| `runs/<experiment>/predictions/` | `test_logits.npy`, `test_predictions.npy`, `test_predictions.csv` |
| `runs/<experiment>/predictions/metrics/` | Per-label and summary metrics from script 010 |
| `outputs/<date>/<time>/` | Hydra logs and CSV metrics (duplicate of fairseq logging; safe to delete) |

## Notes

- **MIMIC checkpoint:** the downloaded `mimic_iv_ecg_finetuned.pt` references a path on the authors' cluster. Used the converted `mimic_iv_ecg_finetuned_encoder.pt` for training (see `06_patch_mimic_checkpoint.py`).
- **Disk usage:** set `checkpoint.keep_last_epochs=1` in training scripts to avoid keeping all 140 epoch checkpoints (~1 GB each).
- **Shared code:** `scripts/ecg_common.py` holds path loading, fairseq inference, and prediction export used by scripts 08–010.
