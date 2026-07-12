"""loaders/squtr.py — SQuTR spoken-query-to-text-retrieval loader (B6-retrieval-tools).

SQuTR (SIGIR-2026) ships as a single ``source_data.zip`` (~21 GB) — read DIRECTLY via
``zipfile`` (random-access member reads; a standard, non-solid zip supports seeking to one
member without decompressing the rest), NEVER extracted to disk. Verified on-disk layout
(2026-07-09, this task):

    source_data.zip
      source_data/<lang>/<subset>/
        corpus.jsonl                              -- [{_id, title, text}]
        queries.jsonl                              -- [{_id, text}]              (text-only)
        queries_with_audio_clean.jsonl             -- [{_id, text, audio}]       (+3 noise levels)
        queries_with_audio_noise_snr_0.jsonl
        queries_with_audio_noise_snr_10.jsonl
        queries_with_audio_noise_snr_20.jsonl
        audio_clean/<audio>                        -- actual .wav files (+3 noise dirs)
        audio_noise_snr_0/<audio>
        audio_noise_snr_10/<audio>
        audio_noise_snr_20/<audio>
        qrels/test.jsonl                           -- [{"query-id", "corpus-id", "score"}]

So audio is real wav files inside the zip (NOT embedded bytes in the jsonl — the jsonl
``"audio"`` field is just the filename, e.g. ``"4641.wav"``, resolved under the matching
``audio_<noise>/`` dir). ``score`` is inconsistently typed across subsets (str for
fiqa/hotpotqa/nq, int for MedicalRetrieval/DuRetrieval) — always cast via ``int(score)``.
Only a ``test`` qrels split ships (no train/dev qrels), so ``split`` only accepts ``"test"``.

6 subsets confirmed on disk (field-verified counts, 2026-07-09):

    subset            lang  queries  corpus     qrels
    fiqa              en     648     57,638     1,706
    hotpotqa          en   7,405  5,233,329    14,810
    nq                en   3,452  2,681,468     4,201
    DuRetrieval       zh   2,000    100,001     9,839
    MedicalRetrieval  zh   1,000    100,999     1,000
    T2Retrieval       zh  22,812    118,605   118,932

× 4 noise levels each (clean / snr20 / snr10 / snr0 — i.e. no noise, and additive noise at
SNR 20/10/0 dB). Stage-1 defaults to ``fiqa`` (small, en) per the survey recipe
(`wiki/survey/2026-07-09-datasets-candidates11.json`), which explicitly steers away from
nq/hotpotqa's multi-million-row corpora for Stage-1 cost reasons.

TWO entry points (a deliberate split, documented here so it isn't mistaken for drift from the
README contract):

  - ``load_squtr(split, n, seed, subset, noise_level)`` — the standard
    ``scripts/loaders`` Row-list contract (registered in ``registry.py`` as ``"squtr"`` so
    ``smoke_all()`` covers it). One Row per query; ``gold`` is the list of qrels
    ``(corpus_id, score)`` pairs for that query — it does NOT load ``corpus.jsonl`` (multi-
    million rows for some subsets), keeping the registry smoke path cheap.
  - ``load_squtr_retrieval(subset, noise_level, n, seed, n_distractors)`` — the
    retrieval-shaped entry point requested for this task: returns
    ``{"queries": [{"qid", "wav_or_audioref", "text"}], "corpus_sample": [...],
    "qrels": [(qid, docid, score), ...]}``, built via ``build_mini_corpus()`` (gold docs +
    deterministic sampled distractors) so a caller can run BEIR-style R@k/nDCG@10 without
    paying the full-corpus cost.

Both share ``_load_subset_data()`` for the metadata reads and materialize query audio lazily
via ``_common.wav_from_bytes`` (single-member zip reads, cached under
``wav_cache_dir("squtr")`` keyed by ``<subset>_<noise_level>_<qid>`` so the two entry points
and repeat calls never re-extract the same query twice).
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders

from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

ZIP_REL = "squtr/source_data.zip"

# subset -> (lang, dirname-inside-zip); dirname == subset name for every confirmed subset.
SUBSETS = {
    "fiqa": "en",
    "hotpotqa": "en",
    "nq": "en",
    "DuRetrieval": "zh",
    "MedicalRetrieval": "zh",
    "T2Retrieval": "zh",
}

# caller-facing noise_level -> zip's internal file/dir suffix (shared by queries_with_audio_*
# and audio_* — same suffix for both).
NOISE_LEVELS = {
    "clean": "clean",
    "snr20": "noise_snr_20",
    "snr10": "noise_snr_10",
    "snr0": "noise_snr_0",
}


def _zip_path() -> Path:
    p = datasets_dir() / ZIP_REL
    if not p.exists():
        raise FileNotFoundError(f"squtr zip not found at {p}")
    return p


def _subset_prefix(subset: str) -> str:
    if subset not in SUBSETS:
        raise ValueError(f"unknown squtr subset {subset!r}; choose one of {sorted(SUBSETS)}")
    return f"source_data/{SUBSETS[subset]}/{subset}"


def _read_jsonl_member(zf: "zipfile.ZipFile", member: str) -> list[dict]:
    import io

    with zf.open(member) as fh:
        return [json.loads(line) for line in io.TextIOWrapper(fh, encoding="utf-8")]


def _select_indices(n_total: int, n: int | None, seed: int) -> list[int]:
    """Deterministic sub-selection: sorted-order slice if n is None/>=total, else a seeded
    without-replacement sample re-sorted ascending (reproducible, order-stable)."""
    import numpy as np

    if n is None or n >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n, replace=False)
    return sorted(int(i) for i in idx)


def _load_subset_data(subset: str, noise_level: str):
    """Return (queries, qrels) for subset x noise_level. Does NOT load corpus.jsonl."""
    if noise_level not in NOISE_LEVELS:
        raise ValueError(f"unknown squtr noise_level {noise_level!r}; choose one of {sorted(NOISE_LEVELS)}")
    prefix = _subset_prefix(subset)
    suffix = NOISE_LEVELS[noise_level]
    with zipfile.ZipFile(_zip_path()) as zf:
        queries = _read_jsonl_member(zf, f"{prefix}/queries_with_audio_{suffix}.jsonl")
        qrels_raw = _read_jsonl_member(zf, f"{prefix}/qrels/test.jsonl")
    qrels = [(r["query-id"], r["corpus-id"], int(r["score"])) for r in qrels_raw]
    return queries, qrels


def _materialize_query_wav(subset: str, noise_level: str, qid: str, audio_filename: str) -> str:
    prefix = _subset_prefix(subset)
    suffix = NOISE_LEVELS[noise_level]
    member = f"{prefix}/audio_{suffix}/{audio_filename}"
    key = f"{subset}_{noise_level}_{qid}"
    cache_dir = wav_cache_dir("squtr")
    cached = cache_dir / f"{key}.wav"
    if cached.exists():
        return str(cached)
    with zipfile.ZipFile(_zip_path()) as zf:
        raw = zf.read(member)
    return wav_from_bytes(raw, key, cache_dir)


def load_squtr(
    split: str = "test",
    n: int | None = None,
    seed: int = SLICE_SEED,
    subset: str = "fiqa",
    noise_level: str = "clean",
) -> list[dict]:
    """Standard scripts/loaders Row-list entry point (registered as ``"squtr"``).

    One Row per sampled query. ``gold`` = the qrels for that query (list of
    ``{"corpus_id", "score"}``) — does NOT load ``corpus.jsonl`` (see module docstring); use
    ``load_squtr_retrieval`` / ``build_mini_corpus`` when a mini text corpus is needed.
    """
    if split != "test":
        raise ValueError(f"squtr only ships a 'test' qrels split (got split={split!r})")

    queries, qrels = _load_subset_data(subset, noise_level)
    queries_sorted = sorted(queries, key=lambda q: q["_id"])
    idx = _select_indices(len(queries_sorted), n, seed)
    sel = [queries_sorted[i] for i in idx]

    qrels_by_qid: dict[str, list[tuple[str, int]]] = {}
    for qid, docid, score in qrels:
        qrels_by_qid.setdefault(qid, []).append((docid, score))

    out = []
    ids = []
    for q in sel:
        qid = q["_id"]
        wav = _materialize_query_wav(subset, noise_level, qid, q["audio"])
        gold = [{"corpus_id": d, "score": s} for d, s in qrels_by_qid.get(qid, [])]
        item_id = f"{subset}|{noise_level}|{qid}"
        ids.append(item_id)
        out.append({
            "wav": wav,
            "gold": gold,
            "meta": {
                "dataset": "squtr", "item_id": item_id, "split": split,
                "subset": subset, "noise_level": noise_level, "text": q["text"],
            },
        })

    freeze_ids("squtr", ids, dataset="squtr", seed=seed,
               extra={"subset": subset, "noise_level": noise_level})
    return out


def load_full_corpus(subset: str = "fiqa") -> list[dict]:
    """Every document in ``subset``'s ``corpus.jsonl`` (2026-07-13, ticket #38 item 1/F'-3
    remediation: the OFFICIAL full corpus -- 57,638 docs for fiqa, see module docstring's counts
    table), sorted by ``_id`` (canonical, deterministic order -- same convention as
    ``load_squtr``/``load_squtr_retrieval``'s own ``sorted(..., key=lambda q: q["_id"])``).

    Reads ONLY the ``corpus.jsonl`` zip member -- NEVER ``queries*.jsonl`` or ``qrels/test.jsonl``.
    This is a hard, load-bearing invariant, not an incidental optimization: a 'full' corpus-mode
    KB build (``kb_batch_build.build_squtr_corpus_source(corpus_mode='full')``) must index the
    corpus INDEPENDENTLY of which queries/qrels exist, so that scoring (which reads qrels
    elsewhere, at eval time) can never leak into which documents get indexed -- see that
    function's own docstring for the query-independence rationale (F'-3, Decision-Log 续26).
    """
    with zipfile.ZipFile(_zip_path()) as zf:
        corpus = _read_jsonl_member(zf, f"{_subset_prefix(subset)}/corpus.jsonl")
    return sorted(corpus, key=lambda d: d["_id"])


def build_mini_corpus(qrels: list[tuple], corpus: list[dict], n_distractors: int, seed: int) -> list[dict]:
    """Deterministic mini-corpus = gold docs referenced by ``qrels`` + sampled distractors.

    ``qrels`` — list of ``(qid, docid, score)`` (as returned by ``_load_subset_data`` /
    ``load_squtr_retrieval``'s qrels). ``corpus`` — full corpus rows (``[{_id, title, text},
    ...]``, e.g. from ``corpus.jsonl``). Gold docs are those whose ``_id`` appears as a
    ``docid`` in ``qrels``; distractors are a seeded without-replacement sample of the
    remaining corpus (sorted by ``_id`` before sampling, per the loader determinism contract).
    Returns gold docs (sorted by ``_id``) followed by the sampled distractors (sorted by
    ``_id``) — order is fully reproducible for a given ``(qrels, corpus, n_distractors, seed)``.
    """
    import numpy as np

    gold_ids = sorted({docid for _, docid, _ in qrels})
    by_id = {d["_id"]: d for d in corpus}
    gold_docs = [by_id[i] for i in gold_ids if i in by_id]

    gold_id_set = set(gold_ids)
    pool = sorted((d for d in corpus if d["_id"] not in gold_id_set), key=lambda d: d["_id"])
    k = min(n_distractors, len(pool))
    rng = np.random.default_rng(seed)
    if k > 0:
        pick = sorted(int(i) for i in rng.choice(len(pool), size=k, replace=False))
        distractors = [pool[i] for i in pick]
    else:
        distractors = []
    return gold_docs + distractors


def load_squtr_retrieval(
    subset: str = "fiqa",
    noise_level: str = "clean",
    n: int | None = None,
    seed: int = SLICE_SEED,
    n_distractors: int = 2000,
) -> dict:
    """Retrieval-shaped entry point: BEIR-style queries + a gold+distractor mini-corpus + qrels.

    Returns ``{"queries": [{"qid", "wav_or_audioref", "text"}], "corpus_sample": [{_id, title,
    text}, ...], "qrels": [(qid, docid, score), ...]}``. ``wav_or_audioref`` is a materialized
    wav path (see ``_materialize_query_wav``) — a real path, never raw bytes, despite the name
    (kept for parity with the task-spec's field name; every squtr query IS a real on-disk wav
    inside the zip, never embedded bytes, so there is no "audioref-only" case in practice).

    NOTE: loads the full ``corpus.jsonl`` for ``subset`` before sampling distractors — cheap
    for fiqa/MedicalRetrieval/DuRetrieval (100k-ish rows) but multi-million rows for
    hotpotqa/nq (see module docstring's counts table); Stage-1 defaults to fiqa for this reason.
    """
    queries, qrels = _load_subset_data(subset, noise_level)
    queries_sorted = sorted(queries, key=lambda q: q["_id"])
    idx = _select_indices(len(queries_sorted), n, seed)
    sel = [queries_sorted[i] for i in idx]
    qids = {q["_id"] for q in sel}
    qrels_sel = [(qid, docid, score) for (qid, docid, score) in qrels if qid in qids]

    with zipfile.ZipFile(_zip_path()) as zf:
        corpus = _read_jsonl_member(zf, f"{_subset_prefix(subset)}/corpus.jsonl")
    corpus_sample = build_mini_corpus(qrels_sel, corpus, n_distractors, seed)

    out_queries = [
        {"qid": q["_id"], "wav_or_audioref": _materialize_query_wav(subset, noise_level, q["_id"], q["audio"]),
         "text": q["text"]}
        for q in sel
    ]
    return {"queries": out_queries, "corpus_sample": corpus_sample, "qrels": qrels_sel}
