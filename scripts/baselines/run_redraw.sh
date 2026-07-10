#!/usr/bin/env bash
# scripts/baselines/run_redraw.sh — disjoint-redraw RERUN orchestrator (2026-07-10, owner ruling
# Decision-Log 续10, "dev/test 全部重抽"). Structurally a direct mirror of run_wave3.sh -- ONE
# gpu_session acquire/serve/release cycle for the whole rerun (single backbone, qwen3-omni-30b-gguf
# ONLY -- the redraw task's frozen scope, see redraw_cells.py's module docstring), batched inference
# (-np 4 -c 16384, --parallel 4), resident server started once rather than per-cell.
#
# This script owns ONLY the GPU-session lifecycle (acquire -> serve up -> iterate -> serve down ->
# release) + log routing; the actual precise rerun-list / archive-then-rerun / checkpoint-skip logic
# lives in redraw_cells.py (not reimplemented here).
#
# *** GPU CONTENTION (task brief): a wave-3 run may still hold scripts/gpu_session.sh's lock when
# this is first invoked. gpu_acquire_polite below waits up to GPU_WAIT_MAX_S (default 3600s = 60 min)
# before giving up, exactly mirroring run_wave1.sh/run_wave3.sh's own polite-wait helper. ***
#
# Usage:
#   bash scripts/baselines/run_redraw.sh --dry-run                       # census + wall-clock estimate, zero GPU/model use
#   bash scripts/baselines/run_redraw.sh                                  # run every cell needing rerun (both splits, all 65 datasets)
#   bash scripts/baselines/run_redraw.sh --dataset aishell-1 --split dev  # ONE cell (task brief step 3 validation)
#   GPU_WAIT_MAX_S=1800 bash scripts/baselines/run_redraw.sh              # shorter GPU-wait ceiling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_SESSION_SH="$SCRIPT_DIR/../gpu_session.sh"

: "${SPEECHRL_DATA_DIR:?SPEECHRL_DATA_DIR must be set (see CLAUDE.md Environment section)}"
DATA="$SPEECHRL_DATA_DIR"
LOG_DIR="$DATA/_repro/redraw_logs"         # orchestration logs live on E:, NOT in git (per-cell
mkdir -p "$LOG_DIR"                        # RESULT jsons still go to the repo's _repro/baselines/,
                                            # written by run_baseline.write_result -- unchanged.

VENV="${SPEECHRL_VENV:-$HOME/.venvs/speechrl}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

DRY_RUN=0
ONLY_SPLIT=""
ONLY_DATASET=""
GPU_WAIT_MAX_S="${GPU_WAIT_MAX_S:-3600}"   # polite-wait ceiling (s) -- 60 min default per task brief

REDRAW_NP="${REDRAW_NP:-4}"
REDRAW_CTX="${REDRAW_CTX:-16384}"
REDRAW_CACHE_RAM="${REDRAW_CACHE_RAM:-8192}"
REDRAW_PARALLEL="${REDRAW_PARALLEL:-4}"
export LLAMA_NP="$REDRAW_NP"
export LLAMA_CTX="$REDRAW_CTX"
export LLAMA_CACHE_RAM="$REDRAW_CACHE_RAM"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --split) ONLY_SPLIT="$2"; shift 2 ;;
    --dataset) ONLY_DATASET="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

py_args=(--parallel "$REDRAW_PARALLEL")
[ -n "$ONLY_SPLIT" ] && py_args+=(--split "$ONLY_SPLIT")
[ -n "$ONLY_DATASET" ] && py_args+=(--dataset "$ONLY_DATASET")

if [ "$DRY_RUN" = 1 ]; then
  python "$SCRIPT_DIR/redraw_cells.py" --census "${py_args[@]}"
  exit 0
fi

# gpu_acquire_polite <name> -- retry `gpu_session.sh acquire <name>` until it succeeds or
# GPU_WAIT_MAX_S elapses. --pid "$$" is REQUIRED (command-substitution subshell pid trap) --
# mirrored verbatim from run_wave1.sh/run_wave3.sh, see gpu_session.sh's 2026-07-09 postmortem.
gpu_acquire_polite() {
  local name="$1" waited=0 err
  while true; do
    if err="$(bash "$GPU_SESSION_SH" acquire "$name" --pid "$$" 2>&1)"; then
      echo "$err"
      return 0
    fi
    if ! printf '%s' "$err" | grep -q "already held"; then
      echo "$err" >&2
      return 1
    fi
    echo "GPU busy, waiting (${waited}s/${GPU_WAIT_MAX_S}s) ... $(printf '%s' "$err" | head -n1)"
    if [ "$waited" -ge "$GPU_WAIT_MAX_S" ]; then
      echo "ERROR: gave up waiting for the GPU lock after ${GPU_WAIT_MAX_S}s" >&2
      return 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

BB="qwen3-omni-30b-gguf"   # the redraw task's ONLY backbone (see redraw_cells.py module docstring)
NAME="redraw-rerun"

# extract_field LINE KEY -- pulls "key=value" out of a space-separated REDRAW_SUMMARY line (see
# redraw_cells.py's run_all). Plain parameter-expansion parsing, no PCRE dependency.
extract_field() {
  local line="$1" key="$2" tok
  for tok in $line; do
    case "$tok" in
      "$key"=*) printf '%s' "${tok#"$key"=}"; return 0 ;;
    esac
  done
  printf '?'
}

LOG="$LOG_DIR/redraw_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "=== redraw: backbone=$BB owner=$NAME log=$LOG parallel=$REDRAW_PARALLEL server='-np $REDRAW_NP -c $REDRAW_CTX --cache-ram $REDRAW_CACHE_RAM' ==="

gpu_acquire_polite "$NAME"
bash "$GPU_SESSION_SH" assert-idle
bash "$GPU_SESSION_SH" serve "$BB" up

set +e
python "$SCRIPT_DIR/redraw_cells.py" --execute "${py_args[@]}" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

bash "$GPU_SESSION_SH" serve "$BB" down
bash "$GPU_SESSION_SH" release "$NAME"

summary_line="$(grep '^REDRAW_SUMMARY ' "$LOG" | tail -n1 || true)"
ran="$(extract_field "$summary_line" ran)"
skipped="$(extract_field "$summary_line" skipped)"
failed="$(extract_field "$summary_line" failed)"
total="$(extract_field "$summary_line" total)"
echo "=== redraw: backbone=$BB summary: ran=$ran skipped=$skipped failed=$failed total=$total (exit status=$status) ==="

if [ "$status" -ne 0 ]; then
  echo "WARN: redraw_cells.py --execute exited $status (failed=$failed cells) -- see $LOG" >&2
fi
echo "=== redraw: backbone=$BB done, GPU released ==="

if [ "$failed" != "?" ] && [ "$failed" -gt 0 ] 2>/dev/null; then
  echo "ERROR: redraw completed but $failed cell(s) failed -- see $LOG" >&2
  exit 1
fi
exit "$status"
