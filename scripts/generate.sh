#!/bin/bash

set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
base_dir="$(cd "${scripts_dir}/.." && pwd -P)"

models_dir="${base_dir}/models"
app_path="${base_dir}/tools/pytorch-examples/word_language_model"
samples_dir="${base_dir}/samples"
venv_python="${base_dir}/venvs/torch3/bin/python3"

data_path="${TRAIN_DATA_PATH:-${app_path}/data/rfc}"
default_dropout="${TRAIN_DROPOUT:-0.5}"
train_seed="${TRAIN_SEED:-42}"
# CLI-overridable checkpoint; default matches train.sh naming (includes seed)
checkpoint="${MODEL_CHECKPOINT:-${models_dir}/model_dp${default_dropout}_s${train_seed}.pt}"
output_file="${SAMPLE_OUTPUT:-${samples_dir}/sample.txt}"
word_count="${SAMPLE_WORDS:-200}"


usage() {
    cat <<'EOF'
Usage: ./scripts/generate.sh [options]

Options:
  --checkpoint PATH    Path to model checkpoint (overrides env MODEL_CHECKPOINT)
  --words N            Number of words to generate (default 200)
  --out PATH           Output file for generated text
  --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) checkpoint="$2"; shift 2 ;;
        --words) word_count="$2"; shift 2 ;;
        --out) output_file="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; usage; exit 1 ;;
    esac
done

# Resolve relative paths against repository root so they remain valid after `cd ${app_path}`.
if [[ "${checkpoint}" != /* ]]; then
    checkpoint="${base_dir}/${checkpoint}"
fi
if [[ "${output_file}" != /* ]]; then
    output_file="${base_dir}/${output_file}"
fi
if [[ "${data_path}" != /* ]]; then
    data_path="${base_dir}/${data_path}"
fi

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
mkdir -p "$(dirname "${output_file}")"

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
