#!/usr/bin/env bash
# scripts/baselines/run_wave1.sh — Wave-1 orchestrator: frozen cell list (see wave1_cells.py) x
# {qwen3-omni-30b-gguf, meralion-2-gguf} x {dev, test}. ONE gpu_session acquire/serve/release
# cycle per backbone (not per cell, not per split) -- avoids repeatedly cold-loading the resident
# llama-server (measured ~4 min per wiki/Inference-Engine-Choice.md) between every dataset.
#
# This script owns ONLY the GPU-session lifecycle (acquire -> serve <model-key> up -> iterate ->
# serve down -> release) and log routing; the actual cell list / checkpoint-skip / run_one calls
# live in wave1_cells.py (not reimplemented here).
#
# Usage:
#   bash scripts/baselines/run_wave1.sh --dry-run                       # schedule + wall-clock estimate, zero GPU/model use
#   bash scripts/baselines/run_wave1.sh                                  # run every pending cell, both backbones, both splits
#   bash scripts/baselines/run_wave1.sh --backbone meralion-2-gguf       # one backbone only
#   bash scripts/baselines/run_wave1.sh --backbone qwen3-omni-30b-gguf --split dev --dataset big-bench-audio
#                                                                         # one cell (validation runs)
#
# Per-cell checkpointing: wave1_cells.is_checkpointed skips any (dataset, backbone, split) whose
# _repro/baselines/<dataset>__<backbone>__<split>.json already has a non-null "aggregate" -- safe
# to re-invoke after a partial run, a crash, or a Ctrl-C.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_SESSION_SH="$SCRIPT_DIR/../gpu_session.sh"

: "${SPEECHRL_DATA_DIR:?SPEECHRL_DATA_DIR must be set (see CLAUDE.md Environment section)}"
DATA="$SPEECHRL_DATA_DIR"
LOG_DIR="$DATA/_repro/wave1_logs"          # orchestration logs live on E:, NOT in git (per-cell
mkdir -p "$LOG_DIR"                        # RESULT jsons still go to the repo's _repro/baselines/,
                                            # written by run_baseline.write_result -- unchanged.

VENV="${SPEECHRL_VENV:-$HOME/.venvs/speechrl}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

DRY_RUN=0
ONLY_BACKBONE=""
ONLY_SPLIT=""
ONLY_DATASET=""
GPU_WAIT_MAX_S="${GPU_WAIT_MAX_S:-1800}"   # polite-wait ceiling (s) before giving up on a busy GPU

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --backbone) ONLY_BACKBONE="$2"; shift 2 ;;
    --split) ONLY_SPLIT="$2"; shift 2 ;;
    --dataset) ONLY_DATASET="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

py_cells_args=()
[ -n "$ONLY_BACKBONE" ] && py_cells_args+=(--backbone "$ONLY_BACKBONE")
[ -n "$ONLY_SPLIT" ] && py_cells_args+=(--split "$ONLY_SPLIT")
[ -n "$ONLY_DATASET" ] && py_cells_args+=(--dataset "$ONLY_DATASET")

if [ "$DRY_RUN" = 1 ]; then
  python "$SCRIPT_DIR/wave1_cells.py" --dry-run "${py_cells_args[@]}"
  exit 0
fi

# gpu_acquire_polite <name> -- retry `gpu_session.sh acquire <name>` until it succeeds or
# GPU_WAIT_MAX_S elapses. Another cooperative owner (e.g. a concurrent agent session) may still
# hold the lock from a prior run; this waits it out instead of failing immediately.
gpu_acquire_polite() {
  local name="$1" waited=0 err
  while true; do
    if err="$(bash "$GPU_SESSION_SH" acquire "$name" 2>&1)"; then
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

BACKBONES=(qwen3-omni-30b-gguf meralion-2-gguf)
[ -n "$ONLY_BACKBONE" ] && BACKBONES=("$ONLY_BACKBONE")

for bb in "${BACKBONES[@]}"; do
  NAME="wave1-$bb"
  LOG="$LOG_DIR/wave1_${bb}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "=== wave1: backbone=$bb owner=$NAME log=$LOG ==="

  gpu_acquire_polite "$NAME"
  bash "$GPU_SESSION_SH" assert-idle
  bash "$GPU_SESSION_SH" serve "$bb" up

  cell_args=(--backbone "$bb" --execute)
  [ -n "$ONLY_SPLIT" ] && cell_args+=(--split "$ONLY_SPLIT")
  [ -n "$ONLY_DATASET" ] && cell_args+=(--dataset "$ONLY_DATASET")

  set +e
  python "$SCRIPT_DIR/wave1_cells.py" "${cell_args[@]}" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  set -e

  bash "$GPU_SESSION_SH" serve "$bb" down
  bash "$GPU_SESSION_SH" release "$NAME"

  if [ "$status" -ne 0 ]; then
    echo "ERROR: wave1_cells.py --backbone $bb exited $status -- see $LOG" >&2
    exit "$status"
  fi
  echo "=== wave1: backbone=$bb done, GPU released ==="
done

echo "=== wave1 done (all requested backbones) ==="
