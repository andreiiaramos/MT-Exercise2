#!/bin/bash
# =============================================================================
# sweep.sh  —  Overnight dropout sweep for MT Exercise 2
#
# Trains one model per dropout value sequentially, logging all output.
# Safe to re-run: skips any model whose checkpoint already exists unless
# --force is passed.
#
# Usage:
#   bash scripts/sweep.sh                  # default 6 dropout values
#   bash scripts/sweep.sh --force          # retrain even if checkpoint exists
#   bash scripts/sweep.sh --epochs 20      # override epoch count for all runs
#   bash scripts/sweep.sh --emsize 250 --nhid 250
# =============================================================================

set -euo pipefail

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
base_dir="$(cd "${scripts_dir}/.." && pwd -P)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DROPOUT_VALUES=(0.0 0.1 0.2 0.3 0.5 0.7)   # 6 values, includes 0.0 (no dropout)
EPOCHS=40
EMSIZE=200
NHID=200
SEED=42
FORCE=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)    FORCE=1; shift ;;
        --epochs)   EPOCHS="$2"; shift 2 ;;
        --emsize)   EMSIZE="$2"; shift 2 ;;
        --nhid)     NHID="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --help)
            cat <<'EOF'
Usage: ./scripts/sweep.sh [options]

Options:
  --force          Retrain even if checkpoint already exists
  --epochs N       Epochs per model (default: 40)
  --emsize N       Embedding size (default: 200)
  --nhid N         Hidden size (default: 200)
  --seed N         Random seed (default: 42)
  --help           Show this help
EOF
            exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
models_dir="${base_dir}/models"
logs_dir="${base_dir}/logs"
sweep_log="${logs_dir}/sweep_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${models_dir}"
mkdir -p "${logs_dir}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo " MT Exercise 2 — Dropout Sweep"
echo "============================================================"
echo " Dropout values : ${DROPOUT_VALUES[*]}"
echo " Epochs         : ${EPOCHS}"
echo " Embedding size : ${EMSIZE}"
echo " Hidden size    : ${NHID}"
echo " Seed           : ${SEED}"
echo " Models dir     : ${models_dir}"
echo " Logs dir       : ${logs_dir}"
echo " Master log     : ${sweep_log}"
echo " Force retrain  : ${FORCE}"
echo "============================================================"
echo ""

TOTAL=${#DROPOUT_VALUES[@]}
COMPLETED=0
SKIPPED=0
FAILED=0
SWEEP_START=$SECONDS

# ---------------------------------------------------------------------------
# Helper: log to both stdout and master sweep log
# ---------------------------------------------------------------------------
log() { echo "$*" | tee -a "${sweep_log}"; }

# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
for dp in "${DROPOUT_VALUES[@]}"; do

    checkpoint="${models_dir}/model_dp${dp}_s${SEED}.pt"
    csv_log="${logs_dir}/log_dp${dp}.csv"
    run_log="${logs_dir}/run_dp${dp}.log"

    log ""
    log "------------------------------------------------------------"
    log " Run $((COMPLETED + SKIPPED + FAILED + 1))/${TOTAL}  |  dropout=${dp}"
    log "------------------------------------------------------------"

    # Skip if checkpoint exists and --force not set
    if [[ -f "${checkpoint}" && "${FORCE}" -eq 0 ]]; then
        log " SKIP: checkpoint already exists at ${checkpoint}"
        log "       Delete it or pass --force to retrain."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    log " Checkpoint : ${checkpoint}"
    log " CSV log    : ${csv_log}"
    log " Run log    : ${run_log}"
    log " Started    : $(date '+%Y-%m-%d %H:%M:%S')"

    RUN_START=$SECONDS

    # Run train.sh, capturing output to per-run log AND stdout
    if "${scripts_dir}/train.sh" \
        --dropout "${dp}" \
        --epochs  "${EPOCHS}" \
        --emsize  "${EMSIZE}" \
        --nhid    "${NHID}" \
        --seed    "${SEED}" \
        --save    "${checkpoint}" \
        --log     "${csv_log}" \
        2>&1 | tee "${run_log}" | tee -a "${sweep_log}"; then

        RUN_ELAPSED=$(( SECONDS - RUN_START ))
        log " Finished   : $(date '+%Y-%m-%d %H:%M:%S')  (${RUN_ELAPSED}s)"
        COMPLETED=$((COMPLETED + 1))
    else
        log " FAILED     : dropout=${dp} — see ${run_log}"
        FAILED=$((FAILED + 1))
        # Continue with remaining values instead of aborting the whole sweep
    fi

done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
SWEEP_ELAPSED=$(( SECONDS - SWEEP_START ))
log ""
log "============================================================"
log " Sweep complete"
log "============================================================"
log " Total time : ${SWEEP_ELAPSED}s  (~$(( SWEEP_ELAPSED / 60 ))m)"
log " Completed  : ${COMPLETED}/${TOTAL}"
log " Skipped    : ${SKIPPED}/${TOTAL}  (checkpoint existed)"
log " Failed     : ${FAILED}/${TOTAL}"
log ""

if [[ "${COMPLETED}" -gt 0 ]]; then
    log " CSV logs available for plotting:"
    for dp in "${DROPOUT_VALUES[@]}"; do
        csv_log="${logs_dir}/log_dp${dp}.csv"
        if [[ -f "${csv_log}" ]]; then
            log "   ${csv_log}"
        fi
    done
    log ""
    log " To plot results, run:"
    log "   python3 ${scripts_dir}/plot_perplexities.py \\"
    # Build the file list for the hint
    csv_files=()
    for dp in "${DROPOUT_VALUES[@]}"; do
        f="${logs_dir}/log_dp${dp}.csv"
        [[ -f "$f" ]] && csv_files+=("$f")
    done
    log "     ${csv_files[*]} \\"
    log "     --out-train plots/train_perplexity.png \\"
    log "     --out-valid plots/valid_perplexity.png"
fi

log "============================================================"