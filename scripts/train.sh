#!/bin/bash

set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
base_dir="$(cd "${scripts_dir}/.." && pwd -P)"

models_dir="${base_dir}/models"
app_path="${base_dir}/tools/pytorch-examples/word_language_model"
default_data_path="${app_path}/data/rfc"
fallback_data_path="${base_dir}/data/rfc/splits"
data_path="${TRAIN_DATA_PATH:-${default_data_path}}"
venv_python="${base_dir}/venvs/torch3/bin/python3"

train_epochs="${TRAIN_EPOCHS:-40}"
train_seed="${TRAIN_SEED:-42}"
num_threads="${OMP_NUM_THREADS:-4}"
device="${CUDA_VISIBLE_DEVICES:-}"
dropout="${TRAIN_DROPOUT:-0.0}"
embedding_size="${TRAIN_EMSIZE:-200}"
hidden_size="${TRAIN_NHID:-200}"
save_path=""
log_path=""
IGNORE_CHECKSUM=0

usage() {
    cat <<'EOF'
Usage: ./scripts/train.sh [options]

Options:
  --data PATH          Dataset directory containing train.txt/valid.txt/test.txt
  --epochs N           Number of training epochs
  --seed N             Random seed
  --dropout FLOAT      Dropout value (e.g. 0, 0.3, 0.5)
  --emsize N           Embedding size
  --nhid N             Hidden layer size
  --threads N          OMP thread count
  --device VALUE       CUDA_VISIBLE_DEVICES value (empty for CPU)
  --save PATH          Output model path (default: models/model_dp<dropout>.pt)
  --log PATH           Write full training output to a log file as well
  --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) data_path="$2"; shift 2 ;;
        --epochs) train_epochs="$2"; shift 2 ;;
        --seed) train_seed="$2"; shift 2 ;;
        --dropout) dropout="$2"; shift 2 ;;
        --emsize) embedding_size="$2"; shift 2 ;;
        --nhid) hidden_size="$2"; shift 2 ;;
        --threads) num_threads="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --save) save_path="$2"; shift 2 ;;
        --log) log_path="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [ -x "${venv_python}" ]; then
    PYTHON_BIN="${venv_python}"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python3"
else
    echo "ERROR: No Python executable found in virtual environment."
    echo "Run: ./scripts/make_virtualenv.sh && ./scripts/install_packages.sh"
    exit 1
fi

mkdir -p "${models_dir}"

echo "------------------------------------------"
echo "Checking paths..."
echo "App Path: ${app_path}"

if [ -z "${TRAIN_DATA_PATH:-}" ] && [ ! -f "${data_path}/train.txt" ] && [ -f "${fallback_data_path}/train.txt" ]; then
    data_path="${fallback_data_path}"
fi

echo "Data Path: ${data_path}"

if [ -z "${save_path}" ]; then
    save_path="${models_dir}/model_dp${dropout}_s${train_seed}.pt"
fi

if [ -n "${log_path}" ]; then
    mkdir -p "$(dirname "${log_path}")"
fi

if [ ! -f "${app_path}/main.py" ]; then
    echo "ERROR: ${app_path}/main.py not found."
    echo "The pytorch/examples repository was not fully installed."
    echo "Run: ./scripts/install_packages.sh"
    exit 1
fi

if [ ! -f "${data_path}/train.txt" ] || [ ! -f "${data_path}/valid.txt" ] || [ ! -f "${data_path}/test.txt" ]; then
    echo "ERROR: train/valid/test files not found in ${data_path}"
    echo "Run: ./bootstrap.sh"
    exit 1
fi

echo "Data found! Starting PyTorch..."
echo "Training with Dropout: ${dropout}"
echo "Embedding size: ${embedding_size}"
echo "Hidden size: ${hidden_size}"
echo "Epochs: ${train_epochs}"
echo "------------------------------------------"
SECONDS=0

train_cmd=(
    "${PYTHON_BIN}" main.py
    --data "${data_path}"
    --epochs "${train_epochs}"
    --seed "${train_seed}"
    --log-interval 100
    --emsize "${embedding_size}" --nhid "${hidden_size}" --dropout "${dropout}" --tied
        --save "${save_path}"
)

if [ -n "${log_path}" ]; then
    (cd "${app_path}" &&
        CUDA_VISIBLE_DEVICES="${device}" OMP_NUM_THREADS="${num_threads}" "${train_cmd[@]}" \
        2>&1 | tee "${log_path}"
    )
else
    (cd "${app_path}" &&
        CUDA_VISIBLE_DEVICES="${device}" OMP_NUM_THREADS="${num_threads}" "${train_cmd[@]}"
    )
fi

echo "------------------------------------------"
echo "Training finished."
echo "Saved model to: ${save_path}"
if [ -n "${log_path}" ]; then
    echo "Saved training log to: ${log_path}"
fi
echo "Time taken: ${SECONDS} seconds"