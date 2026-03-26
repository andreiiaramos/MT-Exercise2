#!/bin/bash

set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
base_dir="$(cd "${scripts_dir}/.." && pwd -P)"
venv_dir="${base_dir}/venvs/torch3"

if ! command -v python3 >/dev/null 2>&1; then
	echo "ERROR: python3 not found. Please install Python 3 first."
	exit 1
fi

mkdir -p "${base_dir}/venvs"

if [ -d "${venv_dir}" ]; then
	echo "Virtual environment already exists at ${venv_dir}."
else
	echo "Creating virtual environment in ${venv_dir}..."
	python3 -m venv "${venv_dir}"
fi

echo "------------------------------------------"
echo "To activate your environment, run:"
echo "source ${venv_dir}/bin/activate"
echo "------------------------------------------"
