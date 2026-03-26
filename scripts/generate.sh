#!/bin/bash

set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
base_dir="$(cd "${scripts_dir}/.." && pwd -P)"

models_dir="${base_dir}/models"
app_path="${base_dir}/tools/pytorch-examples/word_language_model"
samples_dir="${base_dir}/samples"
venv_python="${base_dir}/venvs/torch3/bin/python3"

data_path="${TRAIN_DATA_PATH:-${app_path}/data/rfc}"
checkpoint="${MODEL_CHECKPOINT:-${models_dir}/model.pt}"
output_file="${SAMPLE_OUTPUT:-${samples_dir}/sample.txt}"
word_count="${SAMPLE_WORDS:-200}"

if [ -x "${venv_python}" ]; then
    PYTHON_BIN="${venv_python}"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python3"
else
    echo "ERROR: No Python executable found in virtual environment."
    echo "Run: ./scripts/make_virtualenv.sh && ./scripts/install_packages.sh"
    exit 1
fi

mkdir -p "${samples_dir}"

if [ ! -f "${checkpoint}" ]; then
    echo "ERROR: Model checkpoint not found at ${checkpoint}"
    echo "Did your training finish successfully?"
    exit 1
fi

echo "------------------------------------------"
echo "Generating text from the model..."
(cd "${app_path}" &&
    "${PYTHON_BIN}" generate.py \
        --data "${data_path}" \
        --words "${word_count}" \
        --checkpoint "${checkpoint}" \
        --outf "${output_file}"
)

echo "------------------------------------------"
echo "Generation finished! Here is what your model wrote:"
echo ""
cat "${output_file}"
echo ""
echo "------------------------------------------"
echo "Saved to: ${output_file}"
