#!/usr/bin/env bash
# Fine-tune ECG-FM from the PhysioNet-pretrained checkpoint on PTB-XL subclasses.
#
# Prerequisites:
#   1. Clone/install fairseq-signals into external/fairseq-signals
#   2. Run scripts 01-04 to build labels, waveforms, manifests, and validate data
#   3. Ensure .mat files contain an ``idx`` field aligned with labels/y.npy rows
#
# Optional:
#   --normalize   Not supported (pretrained checkpoints use normalize=false)
set -euo pipefail

PROJECT_ROOT=/media/2TB/ecg_project
PATHS_FILE="$PROJECT_ROOT/configs/paths.yaml"

USE_NORMALIZE=false
for arg in "$@"; do
  if [[ "$arg" == "--normalize" ]]; then
    USE_NORMALIZE=true
  fi
done

read_path() {
    python3 - "$1" <<PY
import sys
sys.path.insert(0, "$PROJECT_ROOT/scripts")
from ecg_common import load_paths
print(load_paths()[sys.argv[1]])
PY
}

FAIRSEQ_SIGNALS_ROOT="$(read_path fairseq_signals_root)"
PRETRAINED_MODEL="$(read_path pretrained_model)"
LABEL_DIR="$(read_path labels_dir)"
LEAD_MEAN_PATH="$(read_path lead_mean_path)"
LEAD_STD_PATH="$(read_path lead_std_path)"
MANIFEST_DIR="$(read_path manifest_dir)"
OUTPUT_DIR="$(read_path output_dir_pretrained)"

mkdir -p "$OUTPUT_DIR"

echo "PROJECT_ROOT:          $PROJECT_ROOT"
echo "FAIRSEQ_SIGNALS_ROOT:  $FAIRSEQ_SIGNALS_ROOT"
echo "PRETRAINED_MODEL:      $PRETRAINED_MODEL"
echo "LABEL_DIR:             $LABEL_DIR"
echo "MANIFEST_DIR:          $MANIFEST_DIR"
echo "OUTPUT_DIR:            $OUTPUT_DIR"
echo "NORMALIZE:             $USE_NORMALIZE"
echo

NORMALIZE_ARGS=()
if [[ "$USE_NORMALIZE" == true ]]; then
  cat <<'EOF'
ERROR: --normalize is not supported for ECG-FM fine-tuning (scripts 05/06).

The released PhysioNet/MIMIC checkpoints were pretrained with task.normalize=false.
fairseq-signals refuses to load them when fine-tuning with --normalize.

Options:
  1. Run without --normalize (gain-corrected waveforms only; matches pretraining):
       bash scripts/05_finetune_ecgfm_pretrained.sh

  2. For z-score experiments, use the random-init baseline (script 08):
       python3 scripts/08_train_transformer_baseline.py --normalize

See README exp_002 notes for comparing split methods at record level without
conflicting with the pretrained encoder normalization settings.
EOF
  exit 1
fi

# ---- sanity checks ----
test -d "$FAIRSEQ_SIGNALS_ROOT" || { echo "Missing FAIRSEQ_SIGNALS_ROOT: $FAIRSEQ_SIGNALS_ROOT"; exit 1; }
test -f "$PRETRAINED_MODEL" || { echo "Missing checkpoint: $PRETRAINED_MODEL"; exit 1; }
test -f "$LABEL_DIR/label_def.csv" || { echo "Missing label_def.csv"; exit 1; }
test -f "$LABEL_DIR/y.npy" || { echo "Missing y.npy"; exit 1; }
test -f "$LABEL_DIR/pos_weight.txt" || { echo "Missing pos_weight.txt"; exit 1; }
test -f "$MANIFEST_DIR/train.tsv" || { echo "Missing train.tsv"; exit 1; }
test -f "$MANIFEST_DIR/valid.tsv" || { echo "Missing valid.tsv"; exit 1; }
test -f "$MANIFEST_DIR/test.tsv" || { echo "Missing test.tsv"; exit 1; }

CONFIG_DIR="$FAIRSEQ_SIGNALS_ROOT/examples/w2v_cmsc/config/finetuning/ecg_transformer"
test -d "$CONFIG_DIR" || { echo "Missing config dir: $CONFIG_DIR"; exit 1; }
test -f "$CONFIG_DIR/diagnosis.yaml" || { echo "Missing diagnosis.yaml"; exit 1; }

TRAIN_ROWS=$(($(wc -l < "$MANIFEST_DIR/train.tsv") - 1))
VALID_ROWS=$(($(wc -l < "$MANIFEST_DIR/valid.tsv") - 1))
TEST_ROWS=$(($(wc -l < "$MANIFEST_DIR/test.tsv") - 1))

if (( TRAIN_ROWS == 0 )); then
  echo "train.tsv has no samples."
  exit 1
fi
if (( VALID_ROWS == 0 )); then
  echo "valid.tsv has no samples. Re-run scripts 02-03 on the full dataset."
  exit 1
fi
if (( TEST_ROWS == 0 )); then
  echo "test.tsv has no samples. Re-run scripts 02-03 on the full dataset."
  exit 1
fi

NUM_LABELS=$(($(wc -l < "$LABEL_DIR/label_def.csv") - 1))
POS_WEIGHT=$(tr -d '[:space:]' < "$LABEL_DIR/pos_weight.txt")

echo "NUM_LABELS:   $NUM_LABELS"
echo "POS_WEIGHT:   $POS_WEIGHT"
echo "TRAIN_ROWS:   $TRAIN_ROWS"
echo "VALID_ROWS:   $VALID_ROWS"
echo "TEST_ROWS:    $TEST_ROWS"
echo

if command -v fairseq-hydra-train >/dev/null 2>&1; then
  FAIRSEQ_TRAIN=(fairseq-hydra-train)
elif [[ -x "$FAIRSEQ_SIGNALS_ROOT/fairseq-hydra-train" ]]; then
  FAIRSEQ_TRAIN=("$FAIRSEQ_SIGNALS_ROOT/fairseq-hydra-train")
else
  echo "fairseq-hydra-train not found in PATH or $FAIRSEQ_SIGNALS_ROOT"
  echo "Install fairseq-signals first, e.g.:"
  echo "  cd $FAIRSEQ_SIGNALS_ROOT && pip install -e ."
  exit 1
fi

export PYTHONPATH="$FAIRSEQ_SIGNALS_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"${FAIRSEQ_TRAIN[@]}" \
    task.data="$MANIFEST_DIR" \
    model.model_path="$PRETRAINED_MODEL" \
    model.num_labels="$NUM_LABELS" \
    optimization.lr='[1e-06]' \
    optimization.max_epoch=140 \
    dataset.batch_size=16 \
    dataset.num_workers=5 \
    dataset.valid_subset=valid \
    dataset.disable_validation=false \
    distributed_training.distributed_world_size=1 \
    distributed_training.find_unused_parameters=True \
    checkpoint.save_dir="$OUTPUT_DIR" \
    checkpoint.save_interval=1 \
    checkpoint.keep_last_epochs=1 \
    checkpoint.best_checkpoint_metric=auroc \
    checkpoint.maximize_best_checkpoint_metric=true \
    common.log_format=csv \
    +task.label_file="$LABEL_DIR/y.npy" \
    +criterion.pos_weight="$POS_WEIGHT" \
    "${NORMALIZE_ARGS[@]}" \
    --config-dir "$CONFIG_DIR" \
    --config-name diagnosis

if [[ ! -f "$OUTPUT_DIR/checkpoint_best.pt" ]]; then
  echo "Training finished but checkpoint_best.pt was not created."
  exit 1
fi

echo "Saved best checkpoint to $OUTPUT_DIR/checkpoint_best.pt"
