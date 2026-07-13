#!/usr/bin/env bash
# scripts/knowledge/build_full_corpus.sh — ticket #38 item 1 launcher (F'-3 remediation,
# Decision-Log 续26): the resumable, batch-checkpointed FULL-corpus embedding + persist run that
# scripts/baselines/run_mock.py's source_name_for(...) fail-closed error (ticket #38 item 3) names
# as the remediation for a missing squtr corpus-side source.
#
# CPU by DEFAULT (glap / omni-embed-nemotron are both kb_embed.EMBEDDERS[...]['needs_server'] ==
# False — no resident llama-server dependency either way), offline (HF_HUB_OFFLINE=1 — every model
# this launcher can select is already fetched under SPEECHRL_DATA_DIR/models per
# docs/datasets.lock.json), python -u (unbuffered — so a detached run's log file shows checkpoint
# progress line-by-line, not only at process exit; see the umbrella CLAUDE.md WSL detached-run
# gotchas note re: buffered stdout on redirected/background runs).
#
# --device cuda (2026-07-13, performance fix): pass --device cuda to run on the GPU ONLY when it is
# confirmed free (CLAUDE.md — the RTX 5090 is otherwise reserved for the resident llama-server /
# other GPU sessions; check gpu_session.sh / pgrep before opting in). --device is passed straight
# through to build_full_corpus.py, which defaults to cpu and raises a clear error if cuda is
# requested but torch.cuda.is_available() is False. CUDA_VISIBLE_DEVICES is only forced empty
# (CPU-only) when --device cuda was NOT requested, so a real --device cuda run isn't silently
# starved of its own GPU by this launcher.
#
# Embedding all 57,638 fiqa corpus docs is a multi-hour CPU job (much faster on GPU) — THIS SCRIPT
# IS NOT RUN AUTOMATICALLY by anything in this repo (no test, no CI, no other launcher calls it);
# it is a separate, deliberately scheduled operator action. Checkpointing every 500 docs
# (build_full_corpus.py's DEFAULT_BATCH_SIZE) means Ctrl-C / a crash / a detached-session drop loses
# at most one in-flight batch — re-running this SAME command resumes from the checkpoint file
# rather than re-embedding from doc 0.
#
# Usage:
#   bash scripts/knowledge/build_full_corpus.sh --embedder glap
#   bash scripts/knowledge/build_full_corpus.sh --embedder omni-embed-nemotron --subset fiqa
#   bash scripts/knowledge/build_full_corpus.sh --embedder glap --device cuda   # GPU confirmed free only
#   bash scripts/knowledge/build_full_corpus.sh --embedder glap --embed-only   # pre-warm checkpoint only
#   bash scripts/knowledge/build_full_corpus.sh --embedder glap --supersede    # archive an existing source
#
# Detached execution (recommended for a run this long):
#   nohup bash scripts/knowledge/build_full_corpus.sh --embedder glap \
#     > "$SPEECHRL_DATA_DIR/_repro/full_corpus_logs/glap_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
#   # then re-invoke the SAME command later to resume if it was interrupted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SPEECHRL_DATA_DIR:?SPEECHRL_DATA_DIR must be set (see CLAUDE.md Environment section)}"

VENV="${SPEECHRL_VENV:-$HOME/.venvs/speechrl}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export HF_HUB_OFFLINE=1

# Detect --device cuda (as either "--device cuda" or "--device=cuda") among the passed-through
# args so CUDA_VISIBLE_DEVICES is only forced empty for the CPU-default case — a real --device cuda
# run must keep the GPU visible. Default (no --device flag at all) stays CPU-only, matching
# build_full_corpus.py's own --device default.
WANT_CUDA=0
prev_arg=""
for arg in "$@"; do
  case "$arg" in
    --device=cuda) WANT_CUDA=1 ;;
    cuda) [ "$prev_arg" = "--device" ] && WANT_CUDA=1 ;;
  esac
  prev_arg="$arg"
done

if [ "$WANT_CUDA" -eq 1 ]; then
  echo "=== build_full_corpus: --device cuda requested — NOT forcing CUDA_VISIBLE_DEVICES='' ==="
else
  export CUDA_VISIBLE_DEVICES=""     # CPU-only default — safe for an unattended run
fi

LOG_DIR="$SPEECHRL_DATA_DIR/_repro/full_corpus_logs"
mkdir -p "$LOG_DIR"

if [ $# -eq 0 ]; then
  echo "Usage: $0 --embedder {glap|omni-embed-nemotron} [--subset fiqa] [--batch-size 500] [--device {cpu|cuda}] [--embed-only] [--supersede]" >&2
  exit 2
fi

echo "=== build_full_corpus: HF_HUB_OFFLINE=1, python -u ==="
exec python -u "$SCRIPT_DIR/build_full_corpus.py" "$@"
