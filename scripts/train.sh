#!/bin/bash

set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
base_dir="$(cd "${scripts_dir}/.." && pwd -P)"

models_dir="${base_dir}/models"
app_path="${base_dir}/tools/pytorch-examples/word_language_model"
data_path="${TRAIN_DATA_PATH:-${app_path}/data/rfc}"
venv_python="${base_dir}/venvs/torch3/bin/python3"

# --- Added the dropout variable here ---
train_epochs="${TRAIN_EPOCHS:-40}"
train_seed="${TRAIN_SEED:-42}"
num_threads="${OMP_NUM_THREADS:-4}"
device="${CUDA_VISIBLE_DEVICES:-}"
dropout="${TRAIN_DROPOUT:-0.5}"

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
echo "Data Path: ${data_path}"

if [ ! -f "${data_path}/train.txt" ] || [ ! -f "${data_path}/valid.txt" ] || [ ! -f "${data_path}/test.txt" ]; then
    echo "ERROR: train/valid/test files not found in ${data_path}"
    echo "Run: ./bootstrap.sh"
    exit 1
fi

echo "Data found! Starting PyTorch..."
echo "Training with Dropout: ${dropout}"
echo "------------------------------------------"
SECONDS=0

# --- Updated the --dropout and --save arguments below ---
(cd "${app_path}" &&
    CUDA_VISIBLE_DEVICES="${device}" OMP_NUM_THREADS="${num_threads}" "${PYTHON_BIN}" main.py \
        --data "${data_path}" \
        --epochs "${train_epochs}" \
        --seed "${train_seed}" \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout "${dropout}" --tied \
        --save "${models_dir}/model_dp${dropout}.pt"
)

echo "------------------------------------------"
echo "Training finished."
echo "Saved model to: ${models_dir}/model_dp${dropout}.pt"
echo "Time taken: ${SECONDS} seconds"