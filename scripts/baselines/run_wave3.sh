#!/usr/bin/env bash
# scripts/baselines/run_wave3.sh — Wave-3 orchestrator: frozen cell list (see wave3_cells.py) x
# {qwen3-omni-30b-gguf ONLY} x {dev, test}. ONE gpu_session acquire/serve/release cycle for the
# whole run (single backbone, unlike wave-1/wave-2's per-backbone loop) -- avoids repeatedly
# cold-loading the resident llama-server (measured ~4 min per wiki/Inference-Engine-Choice.md).
# Structurally a direct mirror of run_wave1.sh -- wave-3 covers K3 (fleurs-r) + K10 (audio2tool) +
# K11 (voicebench-advbench, voicebench-ifeval), 4 dataset keys x 1 backbone x 2 splits = 8 cells,
# see wave3_cells.py.
#
# This script owns ONLY the GPU-session lifecycle (acquire -> serve <model-key> up -> iterate ->
# serve down -> release) and log routing; the actual cell list / checkpoint-skip / run_one calls
# live in wave3_cells.py (not reimplemented here).
#
# *** BATCHED INFERENCE (2026-07-10, owner-approved): the resident server is started with
# -np "$WAVE3_NP" -c "$WAVE3_CTX" (default -np 4 -c 16384, via gpu_session.sh's LLAMA_NP/LLAMA_CTX
# overrides) and wave3_cells.py is invoked with --parallel "$WAVE3_PARALLEL" (default 4) so each
# cell's items are dispatched as concurrent HTTP requests (concurrent.futures.ThreadPoolExecutor in
# run_baseline.run_one) instead of one at a time. Smoke verdict backing this: mtmd-compatible, 0/24
# greedy flips vs sequential, VRAM ~20.9GB, 1.75x wall-clock speedup at K=4 (no prompt-cache reuse).
# --cache-ram is left at gpu_session.sh's default (8192 MiB, still ON) -- unaffected by this change,
# still recorded in each result JSON's sampling_params (see run_baseline.server_slot_config_for). ***
#
# *** NO RELEASE GATE (unlike run_wave2.sh): the owner released wave-3 (including batched
# inference) in the same breath as approving the smoke test -- see wave3_cells.py's module
# docstring. This script is therefore immediately executable, no WAVE3_RELEASE env var required. ***
#
# Usage:
#   bash scripts/baselines/run_wave3.sh --dry-run                       # schedule + wall-clock estimate, zero GPU/model use
#   bash scripts/baselines/run_wave3.sh                                  # run every pending cell (both splits)
#   bash scripts/baselines/run_wave3.sh --split dev --dataset fleurs-r   # one cell (validation run, task brief step 4)
#   WAVE3_PARALLEL=1 bash scripts/baselines/run_wave3.sh                 # force sequential (e.g. to A/B against batched)
#
# Per-cell checkpointing: wave3_cells.is_checkpointed skips any (dataset, backbone, split) whose
# _repro/baselines/<dataset>__<backbone>__<split>.json already has a non-null "aggregate" -- safe
# to re-invoke after a partial run, a crash, or a Ctrl-C.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_SESSION_SH="$SCRIPT_DIR/../gpu_session.sh"

: "${SPEECHRL_DATA_DIR:?SPEECHRL_DATA_DIR must be set (see CLAUDE.md Environment section)}"
DATA="$SPEECHRL_DATA_DIR"
LOG_DIR="$DATA/_repro/wave3_logs"          # orchestration logs live on E:, NOT in git (per-cell
mkdir -p "$LOG_DIR"                        # RESULT jsons still go to the repo's _repro/baselines/,
                                            # written by run_baseline.write_result -- unchanged.

VENV="${SPEECHRL_VENV:-$HOME/.venvs/speechrl}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

DRY_RUN=0
ONLY_SPLIT=""
ONLY_DATASET=""
GPU_WAIT_MAX_S="${GPU_WAIT_MAX_S:-1800}"   # polite-wait ceiling (s) before giving up on a busy GPU

# Wave-3 batched-inference server config (2026-07-10 owner-approved smoke, see header) -- exported
# so BOTH the `serve up` call below (which reads LLAMA_NP/LLAMA_CTX) AND the wave3_cells.py
# --execute call (which reads the SAME env vars back via run_baseline.server_slot_config_for, to
# RECORD what was actually used) see identical values.
WAVE3_NP="${WAVE3_NP:-4}"
WAVE3_CTX="${WAVE3_CTX:-16384}"
WAVE3_CACHE_RAM="${WAVE3_CACHE_RAM:-8192}"     # unchanged default -- --cache-ram stays ON, see header
WAVE3_PARALLEL="${WAVE3_PARALLEL:-4}"           # client-side concurrency; matches WAVE3_NP by default
export LLAMA_NP="$WAVE3_NP"
export LLAMA_CTX="$WAVE3_CTX"
export LLAMA_CACHE_RAM="$WAVE3_CACHE_RAM"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --split) ONLY_SPLIT="$2"; shift 2 ;;
    --dataset) ONLY_DATASET="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

py_cells_args=(--parallel "$WAVE3_PARALLEL")
[ -n "$ONLY_SPLIT" ] && py_cells_args+=(--split "$ONLY_SPLIT")
[ -n "$ONLY_DATASET" ] && py_cells_args+=(--dataset "$ONLY_DATASET")

if [ "$DRY_RUN" = 1 ]; then
  python "$SCRIPT_DIR/wave3_cells.py" --dry-run "${py_cells_args[@]}"
  exit 0
fi

# gpu_acquire_polite <name> -- retry `gpu_session.sh acquire <name>` until it succeeds or
# GPU_WAIT_MAX_S elapses. --pid "$$" is REQUIRED (command-substitution subshell pid trap) --
# mirrored verbatim from run_wave1.sh/run_wave2.sh, see gpu_session.sh's 2026-07-09 postmortem.
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

BB="qwen3-omni-30b-gguf"   # wave-3's ONLY backbone (see wave3_cells.py module docstring)

# extract_field LINE KEY -- pulls "key=value" out of a space-separated WAVE3_SUMMARY line (see
# wave3_cells.py's run_backbone). Plain parameter-expansion parsing, no PCRE dependency. Prints "?"
# if the key/line is absent (e.g. wave3_cells.py crashed before printing a summary at all).
extract_field() {
  local line="$1" key="$2" tok
  for tok in $line; do
    case "$tok" in
      "$key"=*) printf '%s' "${tok#"$key"=}"; return 0 ;;
    esac
  done
  printf '?'
}

NAME="wave3-$BB"
LOG="$LOG_DIR/wave3_${BB}_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "=== wave3: backbone=$BB owner=$NAME log=$LOG parallel=$WAVE3_PARALLEL server='-np $WAVE3_NP -c $WAVE3_CTX --cache-ram $WAVE3_CACHE_RAM' ==="

gpu_acquire_polite "$NAME"
bash "$GPU_SESSION_SH" assert-idle
bash "$GPU_SESSION_SH" serve "$BB" up

cell_args=(--backbone "$BB" --execute "${py_cells_args[@]}")

set +e
python "$SCRIPT_DIR/wave3_cells.py" "${cell_args[@]}" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

bash "$GPU_SESSION_SH" serve "$BB" down
bash "$GPU_SESSION_SH" release "$NAME"

# wave3_cells.py prints one "WAVE3_SUMMARY backbone=... ran=... skipped=... failed=... total=..."
# line (in addition to its human-readable summary) and exits non-zero itself if failed>0 -- surface
# both here too so run_wave3.sh never exits 0 while cells silently failed (mirrors
# run_wave1.sh/run_wave2.sh's 2026-07-09/07-10 postmortem fixes).
summary_line="$(grep '^WAVE3_SUMMARY ' "$LOG" | tail -n1 || true)"
ran="$(extract_field "$summary_line" ran)"
skipped="$(extract_field "$summary_line" skipped)"
failed="$(extract_field "$summary_line" failed)"
total="$(extract_field "$summary_line" total)"
echo "=== wave3: backbone=$BB summary: ran=$ran skipped=$skipped failed=$failed total=$total (exit status=$status) ==="

if [ "$status" -ne 0 ]; then
  echo "WARN: wave3_cells.py --backbone $BB exited $status (failed=$failed cells) -- see $LOG" >&2
fi
echo "=== wave3: backbone=$BB done, GPU released ==="

if [ "$failed" != "?" ] && [ "$failed" -gt 0 ] 2>/dev/null; then
  echo "ERROR: wave3 completed but $failed cell(s) failed -- see $LOG" >&2
  exit 1
fi
exit "$status"
