#!/bin/bash

set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
base_dir="$(cd "${scripts_dir}/.." && pwd -P)"
tools_dir="${base_dir}/tools"
req_file="${base_dir}/requirements.lock.txt"
venv_python="${base_dir}/venvs/torch3/bin/python3"

mkdir -p "${tools_dir}"

if [ -x "${venv_python}" ]; then
    PYTHON_BIN="${venv_python}"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python3"
else
    echo "ERROR: No virtual environment found."
    echo "Run: ./scripts/make_virtualenv.sh"
    exit 1
fi

if [ ! -f "${req_file}" ]; then
    echo "ERROR: Missing dependency lock file at ${req_file}"
    exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Upgrading pip and installing pinned packages..."
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${req_file}"

# Function to safely clone or update git repos
install_repo() {
    local url=$1
    local dir=$2
    if [ -d "$dir" ]; then
        if [ -d "$dir/.git" ]; then
            echo "Updating $(basename "$dir")..."
            if (cd "$dir" && git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1); then
                (cd "$dir" && git pull --ff-only)
            else
                echo "WARN: No upstream tracking branch for $(basename "$dir"); skipping pull."
            fi
        else
            if [ -z "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
                echo "Directory exists but is empty; cloning $(basename "$dir")..."
                rm -rf "$dir"
                git clone "$url" "$dir"
            else
                backup_dir="${dir}.backup.$(date +%Y%m%d%H%M%S)"
                echo "WARN: $dir exists but is not a git repository."
                echo "      Moving it to: $backup_dir"
                mv "$dir" "$backup_dir"
                echo "Cloning $(basename "$dir")..."
                git clone "$url" "$dir"
            fi
        fi
    else
        echo "Cloning $(basename "$dir")..."
        git clone "$url" "$dir"
    fi
}

install_repo "https://github.com/bricksdont/moses-scripts" "${tools_dir}/moses-scripts"
install_repo "https://github.com/pytorch/examples" "${tools_dir}/pytorch-examples"


echo "------------------------------------------"
echo "Applying MT Exercise 2 patch to main.py..."
"${PYTHON_BIN}" "${scripts_dir}/patch_main.py"
echo "------------------------------------------"

echo "Installation complete!"