# ECG fine-tuning project

### By: Tal Noy, Ofek Sapir

---

## Overview

Fine-tune [ECG-FM](https://github.com/bowang-lab/ECG-FM) on PTB-XL **diagnostic subclass** labels using [fairseq-signals](https://github.com/Jwoo5/fairseq-signals). Each 10 s ECG record is cut into **5 s segments at 500 Hz** (2500 samples × 12 leads). The exact cutting strategy is configurable (see [Waveform split methods](#waveform-split-methods)). PTB-XL fold splits are: train 1–8, valid 9, test 10.

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

## Waveform split methods

Script `02_prepare_ptbxl_waveforms.py` extracts fixed **5 s** windows from each **10 s** PTB-XL record. Every segment from the same record gets the **same record-level labels** (from script 01).

| Method | Flag | Windows per 10 s record | Description |
|--------|------|-------------------------|-------------|
| **Two halves** | `two_halves` (default) | 2 | Non-overlapping: 0–5 s, 5–10 s |
| **2.5 s overlap** | `overlap_2p5` | 3 | 0–5 s, 2.5–7.5 s, 5–10 s |
| **1 s sliding** | `overlap_1s` | 6 | 0–5, 1–6, 2–7, 3–8, 4–9, 5–10 s |
| **Random crop** | `random` | 1 | One random 5 s window per record (`--seed` for reproducibility) |

Approximate test-set segment counts (~2,198 records in fold 10):

| Method | Test segments |
|--------|---------------|
| `two_halves` | 4,396 |
| `overlap_2p5` | 6,594 |
| `overlap_1s` | 13,188 |
| `random` | 2,198 |

Each method writes to its **own processed folder** (do not share folders between methods — segment filenames overlap and would corrupt data):

```
data/processed/ptbxl_subclass_{split_method}/
  waveforms/          *.mat segment files
  metadata/           samples.csv, split_config.yaml
  labels/             y.npy, pos_weight.txt, record_labels.csv, ...
  manifests/          train.tsv, valid.tsv, test.tsv
```

### Switching split methods (`configs/paths.yaml`)

Set `split_method` and re-run scripts 01–04 for a new dataset. Derived paths update automatically:

```yaml
split_method: overlap_2p5
processed_root: /media/2TB/ecg_project/data/processed/ptbxl_subclass_{split_method}
```

Training and evaluation scripts (03–010) read only from `paths.yaml` — no `--split-method` flag needed after data prep.

**Prepare a new split method:**

```bash
# 1. Set split_method in configs/paths.yaml
python3 scripts/01_make_ptbxl_labels.py
python3 scripts/02_prepare_ptbxl_waveforms.py --split-method overlap_2p5
python3 scripts/03_make_manifests.py
python3 scripts/04_check_dataset.py
```

Script 02 refuses to overwrite a folder prepared with a different method (use a separate `processed_root`, or `--force` to overwrite intentionally).

### Record-level evaluation (10 s)

Training and inference run on **segments**. To evaluate on full **10 s records**, mean-aggregate segment logits per `ecg_id`:

```bash
python3 scripts/09_predict_ecgfm.py --aggregate-records mean
python3 scripts/010_evaluate_predictions.py \
  --predictions-dir runs/ecgfm_ptbxl_subclass/overlap_2p5/pretrained_exp_001/predictions \
  --aggregate-records mean
```

Default (no flag) = segment-level metrics. With `--aggregate-records mean`, script 09 also writes `record_test_*.npy/csv`; script 010 writes metrics under `predictions/metrics/record/`.

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

### Data prep (example: `two_halves`)

```bash
# split_method: two_halves in configs/paths.yaml
python3 scripts/01_make_ptbxl_labels.py
python3 scripts/02_prepare_ptbxl_waveforms.py --split-method two_halves
python3 scripts/03_make_manifests.py
python3 scripts/04_check_dataset.py
```

### Training

With `split_method` set in `paths.yaml`, run scripts as-is (example for `overlap_2p5`):

```bash
mkdir -p runs/ecgfm_ptbxl_subclass/overlap_2p5/pretrained_exp_001

bash scripts/05_finetune_ecgfm_pretrained.sh 2>&1 \
  | tee runs/ecgfm_ptbxl_subclass/overlap_2p5/pretrained_exp_001/train.log

bash scripts/06_finetune_ecgfm_mimic_finetuned.sh 2>&1 \
  | tee runs/ecgfm_ptbxl_subclass/overlap_2p5/mimic_finetuned_exp_001/train.log

python3 scripts/08_train_transformer_baseline.py 2>&1 \
  | tee runs/ecgfm_ptbxl_subclass/overlap_2p5/transformer_baseline_exp_001/train.log
```

Checkpoints are written to the `output_dir_*` paths in `configs/paths.yaml` (include `{split_method}`). Best model is selected by validation **AUROC** (`checkpoint_best.pt`). Use `checkpoint.keep_last_epochs=1` (already set in 05/06/08) to avoid keeping all 140 epoch files (~1 GB each).

### Inference and evaluation

```bash
python3 scripts/09_predict_ecgfm.py \
  --checkpoint runs/ecgfm_ptbxl_subclass/overlap_2p5/pretrained_exp_001/checkpoint_best.pt

python3 scripts/010_evaluate_predictions.py \
  --predictions-dir runs/ecgfm_ptbxl_subclass/overlap_2p5/pretrained_exp_001/predictions
```

For record-level metrics add `--aggregate-records mean` to both scripts. For the MIMIC experiment, point `--checkpoint` at `mimic_finetuned_exp_001/checkpoint_best.pt`.

## Outputs

| Location | Contents |
|----------|----------|
| `runs/ecgfm_ptbxl_subclass/{split_method}/<experiment>/` | `checkpoint_best.pt`, `train.log` |
| `.../predictions/` | `test_logits.npy`, `test_predictions.npy`, `test_predictions.csv` |
| `.../predictions/record_*` | Record-level predictions (when `--aggregate-records mean` on script 09) |
| `.../predictions/metrics/` | Segment-level metrics from script 010 |
| `.../predictions/metrics/record/` | Record-level metrics (when `--aggregate-records mean` on script 010) |
| `outputs/<date>/<time>/` | Hydra logs and CSV metrics (duplicate of fairseq logging; safe to delete) |

## Notes

- **MIMIC checkpoint:** the downloaded `mimic_iv_ecg_finetuned.pt` references a path on the authors' cluster. Use the converted `mimic_iv_ecg_finetuned_encoder.pt` for training (see `06_patch_mimic_checkpoint.py`).
- **Normalization:** waveforms are **gain-corrected** to physical units (script 02). Per-lead z-score (`--normalize`) is **not supported** on scripts 05/06 because the released ECG-FM checkpoints were pretrained with `normalize=false`. Use `--normalize` on script **08** (random-init baseline) only.
- **Shared code:** `scripts/ecg_common.py` holds path loading (with `{split_method}` expansion), fairseq inference, segment/record aggregation, and prediction export used by scripts 08–010.
- **Compute:** GPU Model was NVIDIA GeForce RTX 3080 Ti with 12 GB total VRAM, with Nvidia driver 535.274.02, supports CUDA 12.2.

## Experiments

- **exp_001:** Test whether MIMIC fine-tuning improves transfer to PTB-XL (`two_halves`, no z-score).
- **exp_002:** Compare split methods and record-level mean aggregation (`--aggregate-records mean`).
- **exp_003:** Frozen ECG-FM encoder vs. fine-tuned on PTB-XL.
- **exp_004:** Label-efficiency tests — 1%, 5%, 10%, 25%, 50%, 100% of the training set.
