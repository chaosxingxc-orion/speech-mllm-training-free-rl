#!/usr/bin/env bash
# scripts/baselines/run_phase_a.sh — Step-2 Phase-A orchestrator (ticket #25 P2f, 2026-07-11): ONE
# gpu_session acquire/serve/release cycle for the single Phase-A backbone (qwen3-omni-30b-gguf,
# owner ruling grid draft §6.6 -- MERaLiON added at Phase B), around a `phase_a_cells.py --execute`
# call. Mirrors `run_wave1.sh`'s exact lifecycle shape (acquire -> assert-idle -> serve up ->
# iterate -> serve down -> release), INCLUDING the `--pid "$$"` command-substitution-subshell fix
# (see gpu_session.sh's 2026-07-09 postmortem comment) -- copied, not reinvented.
#
# This script owns ONLY the GPU-session lifecycle + log routing; the cell list / checkpoint-skip /
# run_one_mock calls live in phase_a_cells.py (not reimplemented here).
#
# Usage:
#   bash scripts/baselines/run_phase_a.sh --dry-run                 # schedule + wall-clock estimate, zero GPU/model use
#   bash scripts/baselines/run_phase_a.sh                            # run every pending Phase-A cell
#   bash scripts/baselines/run_phase_a.sh --dataset squtr             # one dataset only
#   bash scripts/baselines/run_phase_a.sh --dim embedder               # one dimension's arms only
#
# Per-cell checkpointing: phase_a_cells.is_checkpointed skips any cell whose
# _repro/step2_mock/<dataset>__<backbone>__<split>__<confighash>.json already has a non-null
# "aggregate" -- safe to re-invoke after a partial run, a crash, or a Ctrl-C.
#
# PRECONDITION this script does NOT check for you: every (dataset, embedder, key_org, value_org)
# KB source a Phase-A cell needs must already be built (`kb_batch_build.build_one` / the wave-1-
# style batch driver once one exists) -- a cell whose source is missing FAILS (caught by
# phase_a_cells.run_cell, does not abort the sweep) rather than building it on the fly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_SESSION_SH="$SCRIPT_DIR/../gpu_session.sh"
PHASE_A_BACKBONE="qwen3-omni-30b-gguf"   # owner ruling, grid draft §6.6 -- MERaLiON at Phase B

: "${SPEECHRL_DATA_DIR:?SPEECHRL_DATA_DIR must be set (see CLAUDE.md Environment section)}"
DATA="$SPEECHRL_DATA_DIR"
LOG_DIR="$DATA/_repro/phase_a_logs"        # orchestration logs live on E:, NOT in git (per-cell
mkdir -p "$LOG_DIR"                        # RESULT jsons still go to the repo's _repro/step2_mock/,
                                            # written by run_mock.write_result_mock -- unchanged.

VENV="${SPEECHRL_VENV:-$HOME/.venvs/speechrl}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

DRY_RUN=0
ONLY_DATASET=""
ONLY_DIM=""
NO_CHECKPOINT=0
GPU_WAIT_MAX_S="${GPU_WAIT_MAX_S:-1800}"   # polite-wait ceiling (s) before giving up on a busy GPU

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --dataset) ONLY_DATASET="$2"; shift 2 ;;
    --dim) ONLY_DIM="$2"; shift 2 ;;
    --no-checkpoint) NO_CHECKPOINT=1; shift ;;
    -h|--help)
      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

py_cells_args=()
[ -n "$ONLY_DATASET" ] && py_cells_args+=(--dataset "$ONLY_DATASET")
[ -n "$ONLY_DIM" ] && py_cells_args+=(--dim "$ONLY_DIM")

if [ "$DRY_RUN" = 1 ]; then
  python "$SCRIPT_DIR/phase_a_cells.py" --dry-run "${py_cells_args[@]}"
  exit 0
fi

# gpu_acquire_polite <name> -- retry `gpu_session.sh acquire <name>` until it succeeds or
# GPU_WAIT_MAX_S elapses -- copied VERBATIM from run_wave1.sh (see its own comment for the exact
# `--pid "$$"` command-substitution-subshell bug this guards against; not reinvented here).
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

# extract_field LINE KEY -- pulls "key=value" out of a space-separated PHASE_A_SUMMARY line (see
# phase_a_cells.py's run_all), mirroring run_wave1.sh's own extract_field verbatim.
extract_field() {
  local line="$1" key="$2" tok
  for tok in $line; do
    case "$tok" in
      "$key"=*) printf '%s' "${tok#"$key"=}"; return 0 ;;
    esac
  done
  printf '?'
}

NAME="phase-a-$PHASE_A_BACKBONE"
LOG="$LOG_DIR/phase_a_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "=== phase-a: backbone=$PHASE_A_BACKBONE owner=$NAME log=$LOG ==="

gpu_acquire_polite "$NAME"
bash "$GPU_SESSION_SH" assert-idle
bash "$GPU_SESSION_SH" serve "$PHASE_A_BACKBONE" up

cell_args=(--execute)
[ -n "$ONLY_DATASET" ] && cell_args+=(--dataset "$ONLY_DATASET")
[ -n "$ONLY_DIM" ] && cell_args+=(--dim "$ONLY_DIM")
[ "$NO_CHECKPOINT" = 1 ] && cell_args+=(--no-checkpoint)

set +e
python "$SCRIPT_DIR/phase_a_cells.py" "${cell_args[@]}" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

bash "$GPU_SESSION_SH" serve "$PHASE_A_BACKBONE" down
bash "$GPU_SESSION_SH" release "$NAME"

summary_line="$(grep '^PHASE_A_SUMMARY ' "$LOG" | tail -n1 || true)"
ran="$(extract_field "$summary_line" ran)"
skipped="$(extract_field "$summary_line" skipped)"
failed="$(extract_field "$summary_line" failed)"
total="$(extract_field "$summary_line" total)"
echo "=== phase-a: summary: ran=$ran skipped=$skipped failed=$failed total=$total (exit status=$status) ==="

if [ "$status" -ne 0 ]; then
  echo "ERROR: phase_a_cells.py --execute exited $status (failed=$failed cells) -- see $LOG" >&2
  exit 1
fi
echo "=== phase-a done, GPU released ==="
