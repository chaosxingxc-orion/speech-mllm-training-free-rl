"""kb_batch_build — Step-2 batch build pipeline: (embedder, dataset-loader-key, pool-split,
value-spec) -> ONE persisted ``kb_build.build_source`` call, gated by the SAME CLEAN
leakage-verdict enforcement every other kb_build caller goes through (``kb_schema.KBLeakageError``
-- see kb_build.py's module docstring). Reuses the existing infrastructure end to end rather than
duplicating it: ``scripts/loaders/registry.py``'s ``LOADERS`` table for the dataset side,
``kb_embed.EMBEDDERS`` for the embedder side, ``kb_build.build_source`` for the actual persist.

``--plan`` mode prints the step-2 build MATRIX (the "8 content-key embedders" axis --
wiki/2026-07-10-step2-grid-draft.md S2 -- crossed with a small default "main-field" dataset pool
set) WITHOUT building anything. The Phase-A dataset selection itself is an owner sign-off item
(same doc, S6) -- NOT decided here; ``MAIN_FIELD_POOLS`` below is a defensible illustrative
default (one pool per major content sub-family: ASR / ST / reasoning-QA / native-retrieval),
override with ``--dataset <any registry.LOADERS key>`` for anything else.

    python scripts/knowledge/kb_batch_build.py --plan                    # matrix, builds nothing
    python scripts/knowledge/kb_batch_build.py --plan --squtr-preview    # + squtr mini-corpus preview
    python scripts/knowledge/kb_batch_build.py --embedder glap --dataset librispeech --split dev \
        --value-spec text --n 40                                        # actually builds ONE source

Lazy-import discipline (CLAUDE.md): heavy deps (torch/transformers/funasr/...) live inside
``kb_embed``'s per-embedder loaders and inside each ``scripts/loaders`` module's ``load_<name>``,
never at this module's top level, so ``import kb_batch_build`` and ``--plan`` stay cheap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                        # scripts/knowledge -- bare `import kb_build` etc.
sys.path.insert(0, str(HERE.parent / "loaders"))      # scripts/loaders -- bare `import registry`/`squtr`

# The step-2 "8 content-key embedders" axis (wiki/2026-07-10-step2-grid-draft.md S2) -- a NAMED
# subset of kb_embed.EMBEDDERS, since that registry also carries speaker/emotion/style embedders
# (eres2netv2, campplus, emotion2vec-*, wavlm-*, dasheng, sensevoice-small) that are out of scope
# for the content-key H-a/H-b decision this matrix is for.
CONTENT_KEY_EMBEDDERS = [
    "glap", "lco-3b", "lco-7b", "qwen3-omni-own", "sense", "meralion-se2", "clap", "omni-embed-nemotron",
]

# Candidate "main-field" (content-key) dataset pools for the --plan matrix -- ONE pool per major
# content sub-family (ASR / multilingual ST / reasoning-QA MCQ / native spoken-query retrieval),
# each a real key in scripts/loaders/registry.py's LOADERS table. This is a DEFAULT/illustrative
# set, sized to match the grid-draft's "~4 datasets" Phase-A scale -- NOT the frozen Phase-A
# selection (that is an owner sign-off item, step2-grid-draft S6). Pass --dataset to build against
# any other registry.LOADERS key instead.
MAIN_FIELD_POOLS = ["librispeech", "fleurs-r", "mmar", "squtr"]


def _registry():
    import registry  # scripts/loaders/registry.py

    return registry


def build_matrix_plan(embedders: list[str] | None = None, pools: list[str] | None = None) -> list[dict]:
    """The (embedder x main-field pool) candidate matrix -- metadata only, builds nothing."""
    import kb_embed

    registry = _registry()
    embedders = embedders or CONTENT_KEY_EMBEDDERS
    pools = pools or MAIN_FIELD_POOLS
    rows = []
    for emb in embedders:
        meta = kb_embed.EMBEDDERS.get(emb, {})
        for ds in pools:
            rows.append({
                "embedder": emb,
                "embedder_dim": meta.get("dim"),
                "embedder_license": meta.get("license"),
                "embedder_needs_server": meta.get("needs_server", False),
                "embedder_note": meta.get("note"),
                "dataset": ds,
                "dataset_registered": ds in registry.LOADERS,
            })
    return rows


def print_plan(embedders: list[str] | None = None, pools: list[str] | None = None) -> None:
    rows = build_matrix_plan(embedders, pools)
    embedders = embedders or CONTENT_KEY_EMBEDDERS
    pools = pools or MAIN_FIELD_POOLS
    print("=== step-2 build matrix (CANDIDATE plan -- Phase-A dataset selection not frozen; "
          "see module docstring) ===")
    print(f"{len(embedders)} content-key embedders x {len(pools)} main-field dataset pools "
          f"= {len(rows)} cells\n")
    header = f"{'embedder':22s} {'dim':>6s} {'server?':>8s}  {'dataset':16s} {'registered?':>11s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        dim = "?" if r["embedder_dim"] is None else str(r["embedder_dim"])
        print(f"{r['embedder']:22s} {dim:>6s} {str(r['embedder_needs_server']):>8s}  "
              f"{r['dataset']:16s} {str(r['dataset_registered']):>11s}")
    notes = {r["embedder"]: r["embedder_note"] for r in rows if r["embedder_note"]}
    if notes:
        print("\n-- embedder notes --")
        for emb, note in notes.items():
            print(f"  {emb}: {note}")


def plan_squtr_mini_corpus(subset: str = "fiqa", noise_level: str = "clean", n: int | None = 40,
                            n_distractors: int = 200, seed: int = 20260705) -> dict:
    """PLAN-MODE ONLY preview of the squtr mini-corpus value-spec path.

    wiki/2026-07-10-step2-grid-draft.md S2's hybrid BM25+dense RRF retrieval arm ("squtr 文本语料
    侧") needs a TEXT-keyed corpus pool alongside squtr's audio-keyed query side. This wires
    ``loaders/squtr.build_mini_corpus`` (via ``load_squtr_retrieval``, which already calls it) to
    preview the shape a real build would take -- gold docs + sampled distractors -- WITHOUT ever
    calling ``kb_build.build_source``: nothing is persisted, no leakage audit runs. A real build
    would key on TEXT (``key_modality='text'``) since this is squtr's TEXT corpus side, not its
    audio query side (which is what ``build_one``/registry's ``"squtr"`` loader embeds).
    """
    import squtr

    retrieval = squtr.load_squtr_retrieval(subset=subset, noise_level=noise_level, n=n, seed=seed,
                                            n_distractors=n_distractors)
    return {
        "would_build_source": f"squtr-{subset}-{noise_level}__mini-corpus",
        "key_modality": "text", "value_type": "text-fact",
        "n_queries": len(retrieval["queries"]),
        "n_corpus_sample": len(retrieval["corpus_sample"]),
        "n_qrels": len(retrieval["qrels"]),
        "note": ("PLAN ONLY -- kb_build.build_source NOT called; no leakage audit ran; nothing "
                 "persisted. Previews the mini-corpus (gold docs + sampled distractors) "
                 "loaders/squtr.build_mini_corpus would hand to a text-keyed source build for the "
                 "hybrid BM25+dense RRF arm."),
    }


# ---- real build path (not exercised by --plan; used once a config is actually selected) --------

def _gold_string(gold) -> str:
    """Best-effort flatten of a loader Row's ``gold`` field to one string, whatever shape a given
    loader uses for it (bare string / dict / list-of-dict, e.g. squtr's qrels list)."""
    if gold is None:
        return ""
    if isinstance(gold, str):
        return gold
    if isinstance(gold, (list, tuple)):
        return "; ".join(_gold_string(g) for g in gold)
    if isinstance(gold, dict):
        return json.dumps(gold, ensure_ascii=False, sort_keys=True)
    return str(gold)


def _extract_value(row: dict, value_spec: str) -> str:
    """Pull the VALUE payload out of a loader Row for a given ``value_spec``.

    ``'text'`` -- ``row['meta']['text']``, the reference transcript/query text every loader in this
    repo's Row contract carries (per squtr.py's convention: ``meta={..., "text": q["text"]}``).
    ``'gold'`` -- a best-effort string form of ``row['gold']`` via ``_gold_string`` (NOTE: this is
    exactly the field the Information-Boundary Guard's leakage audit exists to check -- see
    ``build_one``, which always runs the audit and scrubs regardless of ``value_spec``).
    Anything else is treated as a literal key to pull out of ``row['meta']``.
    """
    if value_spec == "text":
        return str(row.get("meta", {}).get("text", ""))
    if value_spec == "gold":
        return _gold_string(row.get("gold"))
    return str(row.get("meta", {}).get(value_spec, ""))


def build_one(embedder: str, dataset_loader_key: str, pool_split: str, value_spec: str = "text",
              n: int | None = None, seed: int = 20260705, note: str = "") -> dict:
    """Build ONE persisted source: (embedder, dataset_loader_key, pool_split, value_spec) ->
    ``kb_build.build_source(...)``. The CLEAN leakage-verdict enforcement gate applies exactly as
    it does for every other kb_build caller (``kb_schema.KBLeakageError`` on a confirmed LEAKAGE
    verdict unless ``force_persist`` -- not exposed here on purpose, this is a batch pipeline, not a
    debug escape hatch). The audit ALWAYS runs (``audit_golds`` is always populated from each row's
    ``gold`` field) and ``scrub=True`` always -- a batch build must never silently persist a leak.

    ``dataset_loader_key`` must be a key in ``scripts/loaders/registry.py``'s ``LOADERS`` table.
    ``embedder`` must be a key in ``kb_embed.EMBEDDERS``. ``pool_split``/``n``/``seed`` forward to
    the loader's standard ``load_<name>(split, n, seed) -> list[Row]`` contract.
    """
    import kb_build
    import kb_embed

    registry = _registry()
    if dataset_loader_key not in registry.LOADERS:
        raise KeyError(
            f"kb_batch_build.build_one: {dataset_loader_key!r} not in registry.LOADERS "
            f"({len(registry.LOADERS)} registered). Import failures: {registry.IMPORT_ERRORS or 'none'}"
        )
    if embedder not in kb_embed.EMBEDDERS:
        raise KeyError(f"kb_batch_build.build_one: {embedder!r} not in kb_embed.EMBEDDERS "
                        f"(choose one of {sorted(kb_embed.EMBEDDERS)})")

    rows = registry.LOADERS[dataset_loader_key](split=pool_split, n=n, seed=seed)
    records, audit_golds = [], []
    for r in rows:
        records.append({
            "key_audio_ref": r["wav"],
            "value": _extract_value(r, value_spec),
            "from_item_id": r.get("meta", {}).get("item_id"),
        })
        audit_golds.append(_gold_string(r.get("gold")))

    from kb_schema import VALUE_TYPES

    value_type = value_spec if value_spec in VALUE_TYPES else "text-fact"
    source = f"{dataset_loader_key}__{embedder}__{pool_split}"
    return kb_build.build_source(
        source, dataset_loader_key, revision=None, records=records,
        key_modality="audio", value_type=value_type, embedder=embedder,
        audit_golds=audit_golds, scrub=True,
        note=note or f"kb_batch_build: {dataset_loader_key}/{pool_split} x {embedder} (value_spec={value_spec})",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true", help="print the build matrix; build nothing")
    ap.add_argument("--squtr-preview", action="store_true",
                    help="with --plan: also preview the squtr mini-corpus value-spec path (plan-mode only)")
    ap.add_argument("--embedder", help="a kb_embed.EMBEDDERS key")
    ap.add_argument("--dataset", help="a registry.LOADERS key")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--value-spec", default="text")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()

    if args.plan or not (args.embedder and args.dataset):
        print_plan()
        if args.squtr_preview:
            print("\n=== squtr mini-corpus preview (PLAN ONLY -- nothing built) ===")
            print(json.dumps(plan_squtr_mini_corpus(), indent=2, ensure_ascii=False))
        return 0

    manifest = build_one(args.embedder, args.dataset, args.split, args.value_spec, n=args.n, seed=args.seed)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
