#!/usr/bin/env bash
# scripts/baselines/run_wave2.sh — Wave-2 orchestrator: frozen cell list (see wave2_cells.py) x
# {qwen3-omni-30b-gguf, meralion-2-gguf} x {dev, test}. ONE gpu_session acquire/serve/release
# cycle per backbone (not per cell, not per split) -- avoids repeatedly cold-loading the resident
# llama-server (measured ~4 min per wiki/Inference-Engine-Choice.md) between every dataset.
# Structurally a direct mirror of run_wave1.sh (K1+K2+K8+K9-squtr) -- wave-2 covers K4 (SER) + K5
# (speaker-attribute probes) + K6 (SLU intent) + K7 (SLU slot), 16 dataset keys, see wave2_cells.py.
#
# This script owns ONLY the GPU-session lifecycle (acquire -> serve <model-key> up -> iterate ->
# serve down -> release) and log routing; the actual cell list / checkpoint-skip / run_one calls
# live in wave2_cells.py (not reimplemented here).
#
# *** GPU EXECUTION GATE (2026-07-10): wave-2 is LAUNCH-READY but NOT released. *** Any non-dry-run
# invocation of this script requires WAVE2_RELEASE=1 in the environment (owner release, 波2 放行) --
# see the check right after arg parsing below. --dry-run is always safe (zero GPU/model use) and is
# NOT gated.
#
# Usage:
#   bash scripts/baselines/run_wave2.sh --dry-run                       # schedule + wall-clock estimate, zero GPU/model use
#   WAVE2_RELEASE=1 bash scripts/baselines/run_wave2.sh                                  # run every pending cell, both backbones, both splits
#   WAVE2_RELEASE=1 bash scripts/baselines/run_wave2.sh --backbone meralion-2-gguf       # one backbone only
#   WAVE2_RELEASE=1 bash scripts/baselines/run_wave2.sh --backbone qwen3-omni-30b-gguf --split dev --dataset crema-d
#                                                                         # one cell (validation runs)
#
# Per-cell checkpointing: wave2_cells.is_checkpointed skips any (dataset, backbone, split) whose
# _repro/baselines/<dataset>__<backbone>__<split>.json already has a non-null "aggregate" -- safe
# to re-invoke after a partial run, a crash, or a Ctrl-C.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_SESSION_SH="$SCRIPT_DIR/../gpu_session.sh"

: "${SPEECHRL_DATA_DIR:?SPEECHRL_DATA_DIR must be set (see CLAUDE.md Environment section)}"
DATA="$SPEECHRL_DATA_DIR"
LOG_DIR="$DATA/_repro/wave2_logs"          # orchestration logs live on E:, NOT in git (per-cell
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
      sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

py_cells_args=()
[ -n "$ONLY_BACKBONE" ] && py_cells_args+=(--backbone "$ONLY_BACKBONE")
[ -n "$ONLY_SPLIT" ] && py_cells_args+=(--split "$ONLY_SPLIT")
[ -n "$ONLY_DATASET" ] && py_cells_args+=(--dataset "$ONLY_DATASET")

if [ "$DRY_RUN" = 1 ]; then
  python "$SCRIPT_DIR/wave2_cells.py" --dry-run "${py_cells_args[@]}"
  exit 0
fi

# Owner-release gate -- see module header. Checked HERE (not just inside wave2_cells.py's
# --execute path) so a bad GPU acquire never even happens before this script refuses to proceed.
if [ "${WAVE2_RELEASE:-0}" != "1" ]; then
  echo "ERROR: run_wave2.sh: wave-2 execution requires owner release (波2 放行)." >&2
  echo "       Set WAVE2_RELEASE=1 to confirm the owner has released wave-2 for GPU execution." >&2
  exit 2
fi

# gpu_acquire_polite <name> -- retry `gpu_session.sh acquire <name>` until it succeeds or
# GPU_WAIT_MAX_S elapses. Another cooperative owner (e.g. a concurrent agent session) may still
# hold the lock from a prior run; this waits it out instead of failing immediately.
#
# --pid "$$" is REQUIRED here, not optional: this call sits inside a `$(...)` command substitution,
# which forks a transient subshell to host it. gpu_session.sh acquire's default pid ($PPID) would
# resolve to THAT transient subshell -- which exits the instant this line returns -- not to this
# script's own long-lived pid. Without --pid, every liveness check for the rest of the (possibly
# hours-long) wave-2 run would see a dead pid and report the lock falsely STALE. See the
# 2026-07-09 postmortem comment at the top of scripts/gpu_session.sh (this exact bug, diagnosed
# from a 224-cell run that executed zero cells) -- mirrored verbatim here from run_wave1.sh.
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

BACKBONES=(qwen3-omni-30b-gguf meralion-2-gguf)
[ -n "$ONLY_BACKBONE" ] && BACKBONES=("$ONLY_BACKBONE")

# extract_field LINE KEY -- pulls "key=value" out of a space-separated WAVE2_SUMMARY line (see
# wave2_cells.py's run_backbone). Plain parameter-expansion parsing, no PCRE dependency. Prints "?"
# if the key/line is absent (e.g. wave2_cells.py crashed before printing a summary at all).
extract_field() {
  local line="$1" key="$2" tok
  for tok in $line; do
    case "$tok" in
      "$key"=*) printf '%s' "${tok#"$key"=}"; return 0 ;;
    esac
  done
  printf '?'
}

TOTAL_RAN=0
TOTAL_SKIPPED=0
TOTAL_FAILED=0

for bb in "${BACKBONES[@]}"; do
  NAME="wave2-$bb"
  LOG="$LOG_DIR/wave2_${bb}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "=== wave2: backbone=$bb owner=$NAME log=$LOG ==="

  gpu_acquire_polite "$NAME"
  bash "$GPU_SESSION_SH" assert-idle
  bash "$GPU_SESSION_SH" serve "$bb" up

  cell_args=(--backbone "$bb" --execute)
  [ -n "$ONLY_SPLIT" ] && cell_args+=(--split "$ONLY_SPLIT")
  [ -n "$ONLY_DATASET" ] && cell_args+=(--dataset "$ONLY_DATASET")

  set +e
  WAVE2_RELEASE=1 python "$SCRIPT_DIR/wave2_cells.py" "${cell_args[@]}" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  set -e

  bash "$GPU_SESSION_SH" serve "$bb" down
  bash "$GPU_SESSION_SH" release "$NAME"

  # wave2_cells.py prints one "WAVE2_SUMMARY backbone=... ran=... skipped=... failed=... total=..."
  # line per backbone (in addition to its human-readable summary) and exits non-zero itself if
  # failed>0 -- surface both here too so run_wave2.sh never exits 0 while cells silently failed
  # (mirrors run_wave1.sh's 2026-07-09 postmortem fix).
  summary_line="$(grep '^WAVE2_SUMMARY ' "$LOG" | tail -n1 || true)"
  ran="$(extract_field "$summary_line" ran)"
  skipped="$(extract_field "$summary_line" skipped)"
  failed="$(extract_field "$summary_line" failed)"
  total="$(extract_field "$summary_line" total)"
  echo "=== wave2: backbone=$bb summary: ran=$ran skipped=$skipped failed=$failed total=$total (exit status=$status) ==="
  [ "$ran" != "?" ] && TOTAL_RAN=$((TOTAL_RAN + ran))
  [ "$skipped" != "?" ] && TOTAL_SKIPPED=$((TOTAL_SKIPPED + skipped))
  [ "$failed" != "?" ] && TOTAL_FAILED=$((TOTAL_FAILED + failed))

  if [ "$status" -ne 0 ]; then
    # Policy mirrors run_wave1.sh's 2026-07-10 fix: per-cell failures must NOT block the remaining
    # backbones. Record and continue; the final exit code below still reports total failures.
    echo "WARN: wave2_cells.py --backbone $bb exited $status (failed=$failed cells) -- continuing to next backbone; see $LOG" >&2
  fi
  echo "=== wave2: backbone=$bb done, GPU released ==="
done

echo "=== wave2 done (all requested backbones): ran=$TOTAL_RAN skipped=$TOTAL_SKIPPED failed=$TOTAL_FAILED ==="
if [ "$TOTAL_FAILED" -gt 0 ]; then
  echo "ERROR: wave2 completed but $TOTAL_FAILED cell(s) failed -- see per-backbone logs under $LOG_DIR" >&2
  exit 1
fi
