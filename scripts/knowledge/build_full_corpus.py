"""scripts/knowledge/build_full_corpus.py — resumable, batch-checkpointed embedder for ticket #38
item 1's 'full' corpus_mode (F'-3 remediation, Decision-Log 续26).

Embedding squtr's full ``corpus.jsonl`` (57,638 docs for fiqa) in one uninterrupted process is a
multi-hour CPU job (no GPU server needed for glap/omni-embed-nemotron — both are
``needs_server: False`` in ``kb_embed.EMBEDDERS``) that a operator should be able to detach, crash,
Ctrl-C, or resume across sessions without re-embedding work already done. This module is the
embedding half; ``kb_batch_build.build_squtr_corpus_source(..., corpus_mode="full",
precomputed_keys=...)`` (already implemented) is the persistence half — this module's job is ONLY
to hand it a complete ``{doc_id: vector}`` mapping without re-embedding already-checkpointed rows.

Checkpoint format: ``<checkpoint_dir>/full_corpus__<subset>__<embedder>.npz`` — a single
``np.savez`` of a parallel ``doc_ids`` (object array of str) + ``vectors`` (float32 matrix),
written to a ``.tmp`` sibling and ``os.replace``d into place after every batch (atomic on both
POSIX and Windows) — a crash mid-batch loses at most one in-flight batch, never corrupts the
on-disk checkpoint. Resuming re-loads this file, skips any doc id already present, and only embeds
the remainder — byte-identical continuation regardless of how many times the process is
interrupted and restarted, since ``squtr.load_full_corpus`` itself is a deterministic sort by
``_id`` (see its own docstring) and embedding is a pure function of doc text.

This module NEVER calls ``kb_build.build_source`` (the actual store write) itself when run as a
library — ``main()`` (the CLI / launcher entry, ``build_full_corpus.sh``) does that ONLY when
invoked without ``--embed-only``, and even then the launcher is meant for a deliberate, scheduled,
detached run — not invoked automatically by any test or CI path. No test in this repo calls
``main()`` or writes to the real KB store from here.

Run (CPU, offline, detached-ready — see ``build_full_corpus.sh`` for the exact env/flags):
    python -u scripts/knowledge/build_full_corpus.py --embedder glap --subset fiqa
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))              # scripts/knowledge
LOADERS_DIR = os.path.join(os.path.dirname(HERE), "loaders")    # scripts/loaders -- bare `import squtr`
for _p in (HERE, LOADERS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_BATCH_SIZE = 500


def _default_checkpoint_dir() -> str:
    """Checkpoints live under ``SPEECHRL_DATA_DIR`` (E: drive, never git — see CLAUDE.md's
    Environment section) so a detached run's progress survives independent of the repo checkout
    and independent of ``SPEECHRL_KB_DIR`` (the KB *store*, a separate concern from these
    in-progress embedding checkpoints)."""
    data_dir = os.environ.get("SPEECHRL_DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "_repro", "full_corpus_checkpoints")
    return os.path.join(HERE, "_full_corpus_checkpoints")  # offline/test fallback, never used in prod


def checkpoint_path(embedder: str, subset: str, checkpoint_dir: str | None = None) -> str:
    d = checkpoint_dir or _default_checkpoint_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"full_corpus__{subset}__{embedder}.npz")


def load_checkpoint(path: str) -> dict:
    """Returns ``{doc_id: np.ndarray}`` already embedded and persisted at ``path`` — ``{}`` if no
    checkpoint file exists yet (first run)."""
    import numpy as np

    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    ids = data["doc_ids"]
    vecs = data["vectors"]
    return {str(i): v for i, v in zip(ids, vecs)}


def save_checkpoint(path: str, done: dict) -> None:
    """Atomic (tmp-file + ``os.replace``) checkpoint write — never leaves a half-written file at
    ``path`` even if the process is killed mid-save."""
    import numpy as np

    # np.savez appends ".npz" to any filename argument that doesn't already end with it -- the tmp
    # name must ALSO end in ".npz" (not just start with the checkpoint's own name) or savez would
    # silently write to "<tmp>.npz" instead of "<tmp>", and the os.replace below would then look
    # for a file that was never created.
    tmp = path + ".tmp.npz"
    ids = np.array(list(done.keys()), dtype=object)
    vecs = np.stack(list(done.values())).astype("float32")
    np.savez(tmp, doc_ids=ids, vectors=vecs)
    os.replace(tmp, path)


def embed_full_corpus_checkpointed(
    embedder: str,
    subset: str = "fiqa",
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_dir: str | None = None,
    source_lang: str = "eng_Latn",
    corpus_override: list[dict] | None = None,
    progress: bool = True,
) -> dict:
    """Embeds every document in ``subset``'s full corpus (``squtr.load_full_corpus`` — qrels/
    queries NEVER read, same query-independence invariant as ``kb_batch_build.
    build_squtr_corpus_source(corpus_mode='full')``), resuming from any existing checkpoint,
    checkpointing every ``batch_size`` NEWLY-embedded docs (default 500 per ticket #38 item 1).

    ``corpus_override`` (test-only hook): inject a small fake corpus list (``[{"_id":..., "title":
    ..., "text": ...}, ...]``) instead of loading squtr's real multi-GB zip — lets a test exercise
    the resume/checkpoint logic without the real dataset on disk.

    Returns ``{doc_id: np.ndarray}`` covering EVERY doc in the corpus (old + newly embedded).
    """
    import kb_embed

    if corpus_override is not None:
        corpus = corpus_override
    else:
        import squtr

        corpus = squtr.load_full_corpus(subset)

    path = checkpoint_path(embedder, subset, checkpoint_dir)
    done = load_checkpoint(path)
    corpus_ids = {d["_id"] for d in corpus}
    # drop any stale checkpoint entries for doc ids no longer in the corpus (defensive; a fresh
    # corpus.jsonl revision should never actually shrink, but never silently carry ghost rows).
    done = {k: v for k, v in done.items() if k in corpus_ids}
    todo = [d for d in corpus if d["_id"] not in done]

    if progress:
        print(
            f"[build_full_corpus] embedder={embedder!r} subset={subset!r}: "
            f"{len(corpus)} total docs, {len(done)} already checkpointed (resumed), "
            f"{len(todo)} remaining",
            flush=True,
        )

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        texts = [f"{d.get('title', '')} {d.get('text', '')}".strip() for d in batch]
        _, vecs, _ = kb_embed.embed_text(texts, embedder=embedder, source_lang=source_lang)
        for d, v in zip(batch, vecs):
            done[d["_id"]] = v
        save_checkpoint(path, done)
        if progress:
            print(
                f"[build_full_corpus] checkpointed {len(done)}/{len(corpus)} docs "
                f"(+{len(batch)} this batch) -> {path}",
                flush=True,
            )

    missing = [d["_id"] for d in corpus if d["_id"] not in done]
    if missing:
        raise RuntimeError(
            f"embed_full_corpus_checkpointed: {len(missing)} corpus doc id(s) still missing after "
            f"a full pass (first few: {missing[:5]}) — this should be unreachable; investigate "
            "before persisting a source with a partial precomputed_keys mapping."
        )
    return {d["_id"]: done[d["_id"]] for d in corpus}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Resumable, batch-checkpointed embedding of squtr's FULL corpus.jsonl, then a "
            "kb_batch_build.build_squtr_corpus_source(corpus_mode='full') persist. Run via "
            "build_full_corpus.sh (python -u, HF_HUB_OFFLINE=1, CPU) for a real detached run."
        )
    )
    ap.add_argument("--embedder", required=True, choices=("glap", "omni-embed-nemotron"),
                     help="text-embedding-capable, non-GPU-server token (kb_embed.SQUTR_CORPUS_TEXT_CPU_EMBEDDERS)")
    ap.add_argument("--subset", default="fiqa")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--source-lang", default="eng_Latn", help="only consulted for --embedder glap")
    ap.add_argument("--embed-only", action="store_true",
                     help="stop after the checkpoint is complete; do NOT call build_squtr_corpus_source "
                          "(useful to pre-warm the checkpoint separately from the store write)")
    ap.add_argument("--supersede", action="store_true",
                     help="passed through to build_squtr_corpus_source — archive (never delete) any "
                          "existing same-name source rather than refuse-overwrite")
    args = ap.parse_args(argv)

    keys = embed_full_corpus_checkpointed(
        args.embedder, subset=args.subset, batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir, source_lang=args.source_lang,
    )

    if args.embed_only:
        print(f"[build_full_corpus] --embed-only: checkpoint complete ({len(keys)} docs), "
              "skipping build_squtr_corpus_source persist.", flush=True)
        return 0

    import kb_batch_build as kbb

    result = kbb.build_squtr_corpus_source(
        args.embedder, subset=args.subset, corpus_mode="full",
        precomputed_keys=keys, supersede=args.supersede,
    )
    print(f"[build_full_corpus] build_squtr_corpus_source status={result.get('status')!r} "
          f"n_corpus_docs={result.get('n_corpus_docs')} "
          f"five_axis_audit={result.get('five_axis_audit')}", flush=True)
    return 0 if result.get("status") == "built" else 1


if __name__ == "__main__":
    sys.exit(main())
