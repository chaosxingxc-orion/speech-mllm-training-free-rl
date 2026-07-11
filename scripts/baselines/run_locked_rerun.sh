#!/usr/bin/env bash
# scripts/baselines/run_locked_rerun.sh — LOCKED-HOLDOUT confirmatory rerun orchestrator
# (ticket #26 task 5, design doc wiki/2026-07-11-group-split-statistics-design.md §2).
#
# *****************************************************************************************
# *** PREP ONLY — DO NOT RUN. This script (and locked_rerun_cells.py it calls) is written  ***
# *** and reviewable but was NOT executed by the ticket #26 implementation session. The     ***
# *** actual confirmatory rerun against _repro/LOCKED_HOLDOUT/ manifests is a SEPARATE,     ***
# *** owner-gated step (design doc §2.3's access-control convention: only the FINAL         ***
# *** confirmatory scoring pass may read locked test_ids — running this script IS that      ***
# *** pass, so it must not be invoked casually / as part of iterating on the design).        ***
# *****************************************************************************************
#
# Structurally a direct clone of run_redraw.sh -- ONE gpu_session acquire/serve/release cycle for
# the whole rerun (single backbone, qwen3-omni-30b-gguf ONLY -- owner ruling Decision-Log 续13),
# batched inference (-np 4 -c 16384, --parallel 4), resident server started once rather than
# per-cell. This script owns ONLY the GPU-session lifecycle + log routing; the precise
# run-list / archive-then-run / checkpoint-skip / NEW-artifact-id logic lives in
# locked_rerun_cells.py (not reimplemented here) -- see that module's docstring for the RI-G0 rule
# ("the redraw runner's in-place overwrite pattern must NOT be copied": every cell here writes to
# a BRAND-NEW artifact id, <dataset>__qwen3-omni-30b-gguf__{dev,test}.locked.json, and REFUSES to
# overwrite it if it already exists).
#
# *** GPU CONTENTION: a wave-3/redraw run may still hold scripts/gpu_session.sh's lock when this
# is first invoked. gpu_acquire_polite below waits up to GPU_WAIT_MAX_S (default 3600s = 60 min)
# before giving up, mirroring run_wave1.sh/run_redraw.sh's own polite-wait helper. ***
#
# Usage (once actually run -- NOT by this ticket):
#   bash scripts/baselines/run_locked_rerun.sh --dry-run                        # census + wall-clock estimate, zero GPU/model use
#   bash scripts/baselines/run_locked_rerun.sh                                  # run every LOCKED_HOLDOUT cell needing a run
#   bash scripts/baselines/run_locked_rerun.sh --dataset aishell-1 --split dev  # ONE cell (validation)
#   GPU_WAIT_MAX_S=1800 bash scripts/baselines/run_locked_rerun.sh              # shorter GPU-wait ceiling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_SESSION_SH="$SCRIPT_DIR/../gpu_session.sh"

: "${SPEECHRL_DATA_DIR:?SPEECHRL_DATA_DIR must be set (see CLAUDE.md Environment section)}"
DATA="$SPEECHRL_DATA_DIR"
LOG_DIR="$DATA/_repro/locked_rerun_logs"    # orchestration logs live on E:, NOT in git (per-cell
mkdir -p "$LOG_DIR"                        # RESULT jsons still go to the repo's _repro/baselines/,
                                            # written by locked_rerun_cells.run_cell directly (NOT
                                            # run_baseline.write_result -- see that module's
                                            # locked_result_path/RI-G0 note).

VENV="${SPEECHRL_VENV:-$HOME/.venvs/speechrl}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

DRY_RUN=0
ONLY_SPLIT=""
ONLY_DATASET=""
GPU_WAIT_MAX_S="${GPU_WAIT_MAX_S:-3600}"   # polite-wait ceiling (s) -- 60 min default, mirrors run_redraw.sh

LOCKED_NP="${LOCKED_NP:-4}"
LOCKED_CTX="${LOCKED_CTX:-16384}"
LOCKED_CACHE_RAM="${LOCKED_CACHE_RAM:-8192}"
LOCKED_PARALLEL="${LOCKED_PARALLEL:-4}"
export LLAMA_NP="$LOCKED_NP"
export LLAMA_CTX="$LOCKED_CTX"
export LLAMA_CACHE_RAM="$LOCKED_CACHE_RAM"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --split) ONLY_SPLIT="$2"; shift 2 ;;
    --dataset) ONLY_DATASET="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

py_args=(--parallel "$LOCKED_PARALLEL")
[ -n "$ONLY_SPLIT" ] && py_args+=(--split "$ONLY_SPLIT")
[ -n "$ONLY_DATASET" ] && py_args+=(--dataset "$ONLY_DATASET")

if [ "$DRY_RUN" = 1 ]; then
  python "$SCRIPT_DIR/locked_rerun_cells.py" --census "${py_args[@]}"
  exit 0
fi

# gpu_acquire_polite <name> -- retry `gpu_session.sh acquire <name>` until it succeeds or
# GPU_WAIT_MAX_S elapses. --pid "$$" is REQUIRED (command-substitution subshell pid trap) --
# mirrored verbatim from run_wave1.sh/run_redraw.sh, see gpu_session.sh's 2026-07-09 postmortem.
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

BB="qwen3-omni-30b-gguf"   # the locked-rerun task's ONLY backbone (see locked_rerun_cells.py module docstring)
# 2026-07-11 (dev-half rerun task): overridable via LOCKED_RERUN_NAME so a dev-only invocation can
# acquire the GPU lock under a distinct, self-describing owner string (e.g. "locked-dev-rerun") for
# sibling-session clarity -- default unchanged, so every existing/future full-scope caller is unaffected.
NAME="${LOCKED_RERUN_NAME:-locked-rerun}"

# extract_field LINE KEY -- pulls "key=value" out of a space-separated LOCKED_RERUN_SUMMARY line
# (see locked_rerun_cells.py's run_all). Plain parameter-expansion parsing, no PCRE dependency.
extract_field() {
  local line="$1" key="$2" tok
  for tok in $line; do
    case "$tok" in
      "$key"=*) printf '%s' "${tok#"$key"=}"; return 0 ;;
    esac
  done
  printf '?'
}

LOG="$LOG_DIR/locked_rerun_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "=== locked-rerun: backbone=$BB owner=$NAME log=$LOG parallel=$LOCKED_PARALLEL server='-np $LOCKED_NP -c $LOCKED_CTX --cache-ram $LOCKED_CACHE_RAM' ==="

gpu_acquire_polite "$NAME"
bash "$GPU_SESSION_SH" assert-idle
bash "$GPU_SESSION_SH" serve "$BB" up

set +e
python "$SCRIPT_DIR/locked_rerun_cells.py" --execute "${py_args[@]}" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

bash "$GPU_SESSION_SH" serve "$BB" down
bash "$GPU_SESSION_SH" release "$NAME"

summary_line="$(grep '^LOCKED_RERUN_SUMMARY ' "$LOG" | tail -n1 || true)"
ran="$(extract_field "$summary_line" ran)"
skipped="$(extract_field "$summary_line" skipped)"
failed="$(extract_field "$summary_line" failed)"
total="$(extract_field "$summary_line" total)"
echo "=== locked-rerun: backbone=$BB summary: ran=$ran skipped=$skipped failed=$failed total=$total (exit status=$status) ==="

if [ "$status" -ne 0 ]; then
  echo "WARN: locked_rerun_cells.py --execute exited $status (failed=$failed cells) -- see $LOG" >&2
fi
echo "=== locked-rerun: backbone=$BB done, GPU released ==="

if [ "$failed" != "?" ] && [ "$failed" -gt 0 ] 2>/dev/null; then
  echo "ERROR: locked-rerun completed but $failed cell(s) failed -- see $LOG" >&2
  exit 1
fi
exit "$status"
