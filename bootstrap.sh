#!/bin/bash
# =============================================================================
# bootstrap.sh  —  RFC dataset bootstrap for Machine_Learning workspace
#
# Run this script from the Machine_Learning workspace root.
# It will:
#   0. Verify all required tools are present (preflight checks)
#   1. Locate the workspace root containing scripts/
#   2. Download a random set of foundational RFCs from rfc-editor.org (with set seed for reproducibility)
#   2. Clean raw RFC text (strip page breaks, diagrams, TOC fill lines)
#   3. Tokenise with sacremoses and lowercase
#   4. Apply vocabulary limit (top VOCAB_SIZE words; rest -> <unk>)
#   5. Split into train / valid / test and place them where train.sh expects
#
# Usage:
#   bash bootstrap.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT_OVERRIDE=""
VOCAB_SIZE=5000
SEED=42
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=8

# Parse flags
for arg in "$@"; do
  case "${arg}" in
    --repo-root=*) REPO_ROOT_OVERRIDE="${arg#*=}" ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
abort() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

progress_bar() {
  local current="$1"
  local total="$2"
  local label="$3"
  local width=30
  local filled=0
  local percent=0

  if [[ "${total}" -gt 0 ]]; then
    percent=$(( current * 100 / total ))
    filled=$(( current * width / total ))
  fi

  local empty=$(( width - filled ))
  local bar_filled=""
  local bar_empty=""

  if [[ "${filled}" -gt 0 ]]; then
    bar_filled="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  fi
  if [[ "${empty}" -gt 0 ]]; then
    bar_empty="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  fi

  printf "\r%s [%s%s] %3d%% (%d/%d)" "${label}" "${bar_filled}" "${bar_empty}" "${percent}" "${current}" "${total}"

  if [[ "${current}" -ge "${total}" ]]; then
    printf "\n"
  fi
}

# ---------------------------------------------------------------------------
# STAGE 0 — Preflight checks
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Machine_Learning — RFC dataset bootstrap"
echo "============================================================"
echo ""
info "Stage 0/6: Preflight checks"

# --- git ---
if ! command -v git &>/dev/null; then
  abort "'git' not found. Install git and re-run.
         Ubuntu/Debian : sudo apt install git
         macOS         : xcode-select --install  (or brew install git)"
fi
GIT_VERSION=$(git --version)
info "  git     : ${GIT_VERSION}"

# --- curl ---
if ! command -v curl &>/dev/null; then
  abort "'curl' not found. Install curl and re-run.
         Ubuntu/Debian : sudo apt install curl
         macOS         : brew install curl"
fi
CURL_VERSION=$(curl --version | head -1)
info "  curl    : ${CURL_VERSION}"


if ! command -v python3 &>/dev/null; then
  abort "'python3' not found. Install Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} and re-run.
         Ubuntu/Debian : sudo apt install python3 python3-pip
         macOS         : brew install python3
         Windows/WSL   : https://www.python.org/downloads/"
fi

# Check minimum Python version
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [[ "${PY_MAJOR}" -lt "${MIN_PYTHON_MAJOR}" ]] || \
   [[ "${PY_MAJOR}" -eq "${MIN_PYTHON_MAJOR}" && "${PY_MINOR}" -lt "${MIN_PYTHON_MINOR}" ]]; then
  abort "Python ${PY_VERSION} found, but >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} is required.
         Please upgrade Python and re-run."
fi
info "  python3 : ${PY_VERSION}"

# --- pip (warn only — install_packages.sh handles the venv) ---
if ! python3 -m pip --version &>/dev/null 2>&1; then
  warn "pip not found. The virtualenv setup step (make_virtualenv.sh) may fail.
        Ubuntu/Debian : sudo apt install python3-pip"
fi

echo ""
info "All required tools present."
echo ""

# ---------------------------------------------------------------------------
# STAGE 1 — Locate existing workspace root
# ---------------------------------------------------------------------------
info "Stage 1/6: Locate workspace root"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
INVOCATION_DIR="$(pwd)"

if [[ -n "${REPO_ROOT_OVERRIDE}" ]]; then
  if [[ -d "${REPO_ROOT_OVERRIDE}" && -d "${REPO_ROOT_OVERRIDE}/scripts" ]]; then
    REPO_ROOT="${REPO_ROOT_OVERRIDE}"
  else
    warn "Ignoring invalid --repo-root='${REPO_ROOT_OVERRIDE}' (missing scripts/)."
  fi
fi

if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -d "${INVOCATION_DIR}/scripts" ]]; then
    REPO_ROOT="${INVOCATION_DIR}"
  elif [[ -d "${SCRIPT_DIR}/scripts" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
  elif [[ -d "${SCRIPT_DIR}/mt-exercise-02/scripts" ]]; then
    REPO_ROOT="${SCRIPT_DIR}/mt-exercise-02"
  else
    abort "Could not find workspace root with scripts/. Use --repo-root=/absolute/path/to/Machine_Learning"
  fi
fi

if [[ ! -d "${REPO_ROOT}" || ! -d "${REPO_ROOT}/scripts" ]]; then
  abort "Invalid REPO_ROOT: '${REPO_ROOT}' (missing scripts directory)"
fi

info "Using existing repo at: ${REPO_ROOT}"

echo ""

# ---------------------------------------------------------------------------
# Derived paths (all anchored to REPO_ROOT)
# ---------------------------------------------------------------------------
DATA_DIR="${REPO_ROOT}/data/rfc"
RAW_DIR="${DATA_DIR}/raw"
SPLIT_DIR="${DATA_DIR}/splits"
# PyTorch word_language_model default data path (mirrored when available)
TOOLS_DIR="${REPO_ROOT}/tools/pytorch-examples/word_language_model/data/rfc"

CLEANED_FILE="${DATA_DIR}/rfc_cleaned.txt"
TOKENIZED_FILE="${DATA_DIR}/rfc_tokenized.txt"
UNK_FILE="${DATA_DIR}/rfc_unk.txt"

mkdir -p "${RAW_DIR}"
mkdir -p "${SPLIT_DIR}"

# ---------------------------------------------------------------------------
# STAGE 2 — Download RFCs
# ---------------------------------------------------------------------------
info "Stage 2/6: Download RFCs"

# Curated list: foundational networking protocols
RFC_LIST=(1)
# Generate 200 random RFC numbers between 700 and 5000
if command -v shuf >/dev/null 2>&1; then
  while IFS= read -r rfc_num; do
    RFC_LIST+=("${rfc_num}")
  done < <(shuf -i 700-5000 -n 200)
else
  while IFS= read -r rfc_num; do
    RFC_LIST+=("${rfc_num}")
  done < <(SEED="${SEED}" python3 - <<'PY'
import os
import random

seed = int(os.environ.get("SEED", "42"))
random.seed(seed)
nums = random.sample(range(700, 5001), 200)
for n in nums:
    print(n)
PY
  )
fi

rfc_list_sorted="$(printf '%s\n' "${RFC_LIST[@]}" | sort -nu)"
RFC_LIST=()
while IFS= read -r rfc_num; do
  [ -n "${rfc_num}" ] && RFC_LIST+=("${rfc_num}")
done <<< "${rfc_list_sorted}"

info "Fetching up to ${#RFC_LIST[@]} RFCs -> ${RAW_DIR}"
info "(Already-downloaded files are skipped — safe to re-run)"

FETCHED=0
SKIPPED=0
TOTAL_CANDIDATES=${#RFC_LIST[@]}
CURRENT_RFC=0

for rfc_num in "${RFC_LIST[@]}"; do
  CURRENT_RFC=$((CURRENT_RFC + 1))
  progress_bar "${CURRENT_RFC}" "${TOTAL_CANDIDATES}" "Downloading RFCs"

  out_file="${RAW_DIR}/rfc${rfc_num}.txt"

  # Idempotent: skip if already on disk and non-empty
  if [[ -f "${out_file}" && -s "${out_file}" ]]; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  url="https://www.rfc-editor.org/rfc/rfc${rfc_num}.txt"
  curl_stderr_file="$(mktemp)"
  if ! http_code=$(curl --silent \
                       --output "${out_file}" \
                       --write-out "%{http_code}" \
                       --connect-timeout 10 \
                       --max-time 30 \
                       "${url}" 2>"${curl_stderr_file}"); then
    curl_exit_code=$?
    http_code="000"
    curl_error_msg="$(<"${curl_stderr_file}")"
    info "curl failed for ${url} with exit code ${curl_exit_code}: ${curl_error_msg}"
  fi
  rm -f "${curl_stderr_file}"

  if [[ "${http_code}" != "200" ]]; then
    rm -f "${out_file}"
    SKIPPED=$((SKIPPED + 1))
  else
    FETCHED=$((FETCHED + 1))
  fi
done

if [[ "${TOTAL_CANDIDATES}" -eq 0 ]]; then
  echo ""
fi

TOTAL_ON_DISK=$(find "${RAW_DIR}" -name 'rfc*.txt' | wc -l | tr -d ' ')
info "  This run — fetched: ${FETCHED}   not available: ${SKIPPED}"
info "  Total RFC files on disk: ${TOTAL_ON_DISK}"

if [[ "${TOTAL_ON_DISK}" -eq 0 ]]; then
  abort "No RFC files were downloaded. Check your internet connection and re-run."
fi

echo ""

# ---------------------------------------------------------------------------
# STAGE 3 — Clean raw RFC text
# ---------------------------------------------------------------------------
info "Stage 3/6: Clean raw RFC text -> ${CLEANED_FILE}"

python3 - "${RAW_DIR}" "${CLEANED_FILE}" << 'PYEOF'
import sys
import os
import re

raw_dir  = sys.argv[1]
out_path = sys.argv[2]

def is_noise_line(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False   # keep blank lines as paragraph separators

    if '\x0c' in line:
        return True    # form-feed / page break character

    if re.search(r'\[Page\s+\d+\]', stripped):
        return True    # RFC footer: "Author Name    [Page 12]"

    if re.fullmatch(r'[-=_.]{5,}', stripped):
        return True    # separator line

    dot_ratio = stripped.count('.') / max(len(stripped), 1)
    if dot_ratio > 0.4 and len(stripped) > 10:
        return True    # TOC fill line:  "3.1  Overview ........ 7"

    alpha = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 15 and alpha / len(stripped) < 0.35:
        return True    # ASCII diagram or bit-field drawing

    return False

files = sorted(
    f for f in os.listdir(raw_dir)
    if f.startswith('rfc') and f.endswith('.txt')
)

total_files = len(files)
print(f"  Cleaning {total_files} RFC files...", flush=True)

lines_written = 0
with open(out_path, 'w', encoding='utf-8', errors='replace') as out:
  for idx, fname in enumerate(files, start=1):
    fpath = os.path.join(raw_dir, fname)
    try:
      with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    except OSError as e:
      print(f"  WARNING: could not read {fpath}: {e}", file=sys.stderr)
      continue

    for line in lines:
      if not is_noise_line(line):
        out.write(line.rstrip() + '\n')
        lines_written += 1

    out.write('\n')   # blank line between RFC documents

    if idx % 25 == 0 or idx == total_files:
      print(f"  Cleaning progress: {idx}/{total_files} files", flush=True)

print(f"  Lines after cleaning: {lines_written}")
PYEOF

echo ""

# ---------------------------------------------------------------------------
# STAGE 4 — Tokenise with sacremoses
# ---------------------------------------------------------------------------
info "Stage 4/6: Tokenise -> ${TOKENIZED_FILE}"

# sacremoses is installed by install_packages.sh into the venv.
# If this check fails, set up the venv first:
#   ./scripts/make_virtualenv.sh
#   source venvs/torch3/bin/activate
#   ./scripts/install_packages.sh
# Then re-run this script with the venv active.
if ! python3 -c "import sacremoses" 2>/dev/null; then
  abort "'sacremoses' Python package not found.
         Activate the virtualenv before running this script:
           cd ${REPO_ROOT}
           ./scripts/make_virtualenv.sh
           source venvs/torch3/bin/activate
           ./scripts/install_packages.sh
         Then re-run: bash ${INVOCATION_DIR}/bootstrap.sh"
fi

python3 - "${CLEANED_FILE}" "${TOKENIZED_FILE}" << 'PYEOF'
import sys
from sacremoses import MosesTokenizer

in_path  = sys.argv[1]
out_path = sys.argv[2]

tok = MosesTokenizer(lang='en')
lines_written = 0
lines_seen = 0

with open(in_path, 'r', encoding='utf-8', errors='replace') as fin, \
   open(out_path, 'w', encoding='utf-8') as fout:
  for line in fin:
    lines_seen += 1
    line = line.strip()
    if line:
      tokenized = tok.tokenize(
        line.lower(),
        aggressive_dash_splits=True,
        return_str=True
      )
      fout.write(tokenized + '\n')
      lines_written += 1
    else:
      fout.write('\n')

    if lines_seen % 20000 == 0:
      print(f"  Tokenization progress: {lines_seen} input lines", flush=True)

print(f"  Lines tokenised: {lines_written}")
PYEOF

echo ""

# ---------------------------------------------------------------------------
# STAGE 5 — Vocabulary limit: replace rare words with <unk>
# ---------------------------------------------------------------------------
info "Stage 5/6: Apply frequency vocabulary limit (top ${VOCAB_SIZE}) -> ${UNK_FILE}"

python3 - "${TOKENIZED_FILE}" "${UNK_FILE}" "${VOCAB_SIZE}" << 'PYEOF'
import sys
from collections import Counter

in_path    = sys.argv[1]
out_path   = sys.argv[2]
vocab_size = int(sys.argv[3])

counter = Counter()
lines_seen = 0

with open(in_path, 'r', encoding='utf-8') as f:
  for raw_line in f:
    lines_seen += 1
    counter.update(raw_line.split())

    if lines_seen % 50000 == 0:
      print(f"  Vocabulary pass progress: {lines_seen} lines", flush=True)

kept = {word for word, _ in counter.most_common(vocab_size)}
print(f"  Unique tokens in corpus : {len(counter)}")
print(f"  Kept in vocabulary      : {len(kept)}  (remainder -> <unk>)")

unk_count = 0
lines_written = 0
with open(in_path, 'r', encoding='utf-8') as fin, \
   open(out_path, 'w', encoding='utf-8') as fout:
  for line in fin:
    tokens = line.split()
    if tokens:
      replaced = [t if t in kept else '<unk>' for t in tokens]
      unk_count += replaced.count('<unk>')
      fout.write(' '.join(replaced) + '\n')
    else:
      fout.write('\n')

    lines_written += 1
    if lines_written % 50000 == 0:
      print(f"  UNK replacement progress: {lines_written} lines", flush=True)

print(f"  Tokens replaced with <unk>: {unk_count}")
PYEOF

echo ""

# ---------------------------------------------------------------------------
# STAGE 6 — Train / valid / test split (80 / 10 / 10)
# ---------------------------------------------------------------------------
info "Stage 6/6: Split into train / valid / test -> ${SPLIT_DIR}"

python3 - "${UNK_FILE}" "${SPLIT_DIR}" "${SEED}" << 'PYEOF'
import sys
import os
import random

in_path = sys.argv[1]
out_dir = sys.argv[2]
seed    = int(sys.argv[3])

random.seed(seed)

with open(in_path, 'r', encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f if l.strip()]

if len(lines) < 100:
    print(f"ERROR: only {len(lines)} non-empty lines — dataset too small to split.", file=sys.stderr)
    sys.exit(1)

random.shuffle(lines)

n_total = len(lines)
n_train = int(n_total * 0.80)
n_valid = int(n_total * 0.10)
# test receives the remainder

splits = {
    'train.txt': lines[:n_train],
    'valid.txt': lines[n_train : n_train + n_valid],
    'test.txt' : lines[n_train + n_valid :],
}

for fname, data in splits.items():
    path = os.path.join(out_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data) + '\n')
    print(f"  {fname:12s}: {len(data):>6} lines  ->  {path}")
PYEOF

if [[ -f "${REPO_ROOT}/tools/pytorch-examples/word_language_model/main.py" ]]; then
  mkdir -p "${TOOLS_DIR}"
  cp "${SPLIT_DIR}/"*.txt "${TOOLS_DIR}/"
  info "Mirrored split files to: ${TOOLS_DIR}"
else
  warn "pytorch/examples not installed yet; skipped mirroring to ${TOOLS_DIR}."
  warn "Run ./scripts/install_packages.sh to install it."
fi

echo ""
echo "============================================================"
echo " Dataset summary"
echo "============================================================"
wc -l "${SPLIT_DIR}"/*.txt
echo ""
info "Data ready at: ${SPLIT_DIR}"
if [[ -d "${TOOLS_DIR}" ]]; then
  info "PyTorch data mirror: ${TOOLS_DIR}"
fi
echo ""
echo "Next steps:"
echo "  cd ${REPO_ROOT}"
echo "  ./scripts/make_virtualenv.sh"
echo "  source venvs/torch3/bin/activate"
echo "  ./scripts/install_packages.sh"
echo "  ./scripts/train.sh"
