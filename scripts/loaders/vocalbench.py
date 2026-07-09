"""vocalbench — VocalBench (Apache-2.0) axis-level parquet loaders (task B4-recovery).

Recipe: ``wiki/survey/2026-07-09-datasets-lock-second14.md``'s "vocalbench" section -- 4 of
VocalBench's 9 axes have a discrete/verifiable reference and are INCLUDED here:

  knowledge   (2000 rows) -- short-fact spoken QA, free-text gold answer; the survey also flags
              this axis as a top-tier KB-retrieval SOURCE candidate for this dataset family.
  reasoning   (1000 rows) -- short spoken reasoning QA, free-text gold answer.
  multi_round (400 rows)  -- a prior text conversation (``Context``, ``{"from": [...], "value":
              [...]}``) followed by ONE new spoken follow-up question (``audio`` renders
              ``Question``); flattened into ``meta["context"]`` as inline text so a caller can run
              it single-turn (text context + audio question -> gold), per the survey's "Context
              为内联文本可单轮跑" note. No instruction/prompt wording is added -- a plain
              "<role>: <text>" join only (mirrors uro_bench.py's ``_parse_mcq_options``
              precedent: best-effort convenience shaping, never scoring/prompt logic).
  emotion     (500 rows)  -- ``Question_emo`` is a 5-way closed emotion label (angry/happy/
              neutral/sad/surprised, exactly 100 rows each, verified 2026-07-09) attached to a
              spoken, emotionally-loaded question -- re-cast as an SER classification task per
              the survey recipe (mirrors uro_bench.py's UnderEmotion-{en,zh} precedent: gold =
              the label column). This axis has no free-text ``Answer`` column (unlike
              knowledge/reasoning/multi_round) -- only a ``Score`` (list[int], 5 raw human
              intensity ratings) side column, kept in meta for reference only.

Excluded (per the survey recipe, not reconsidered here): creativity / instruction / robust_*
(no reference), safety (judged). ``single_round`` is ALSO excluded -- not named in the survey's
include list, despite superficially looking like another qa-shaped axis.

On-disk layout: ``datasets/vocalbench/parquet/<axis>.parquet`` -- one file per axis, a single
"test" split per the upstream HF dataset card (verified 2026-07-09). Every axis shares a ``Qid``
(str, e.g. "knowledge-0000") and an ``audio`` column (``{"bytes": ..., "path": None}`` -- ``path``
is always ``None`` on this box, so every row is materialized from embedded bytes, mirroring
``voicebench.py``'s convention of never joining a literal path).

Mirrors ``scripts/p2_baselines.py``'s ``load_vocalbench_zh``/``_qa_row`` conventions (audio + a
short free-text answer column -> Row), generalized to VocalBench's (EN) axis-level parquet layout
and this package's ``(split, n, seed) -> list[Row]`` contract -- not reimplemented from scratch.

Row shape (uniform across knowledge/reasoning/multi_round; emotion differs only in gold/meta):
    wav  -> materialized wav path (``audio["bytes"]`` decoded via ``_common.wav_from_bytes``)
    gold -> str: ``Answer`` for knowledge/reasoning/multi_round; ``Question_emo`` for emotion
    meta -> {"dataset": "vocalbench-<axis>", "item_id": Qid, "split": split, "task": <tag>,
             "question": Question, [axis-specific extra fields, see each loader's docstring]}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402


def _require_test_split(dataset: str, split: str) -> None:
    if split != "test":
        raise NotImplementedError(
            f"{dataset}: split={split!r} not wired up -- every VocalBench axis parquet ships a "
            "single 'test' split only (see the upstream HF dataset card)."
        )


def _flatten_context(context) -> str:
    """Best-effort ``(role: text)`` join of a multi_round ``Context`` dict into ONE inline text
    block. Metadata convenience only (mirrors uro_bench.py's ``_parse_mcq_options`` precedent) --
    plain "<role>: <text>" lines, no instruction wording, no reward/prompt logic added.
    """
    if not isinstance(context, dict):
        return str(context)
    froms = context.get("from") or []
    values = context.get("value") or []
    return "\n".join(f"{r}: {v}" for r, v in zip(froms, values))


def _sample_indices(rng, n_total: int, n: int | None):
    import numpy as np

    idx = rng.permutation(n_total) if n is not None else np.arange(n_total)
    return idx[:n] if n is not None else idx


def _load_axis(
    axis: str,
    *,
    gold_col: str = "Answer",
    task: str = "qa",
    meta_cols: tuple = (),
    split: str = "test",
    n: int | None = None,
    seed: int = SLICE_SEED,
) -> list:
    """Shared loader body for knowledge/reasoning/multi_round/emotion -- see module docstring."""
    import numpy as np
    import pyarrow.parquet as pq

    dataset = f"vocalbench-{axis}"
    _require_test_split(dataset, split)

    f = datasets_dir() / "vocalbench" / "parquet" / f"{axis}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{dataset}: no parquet at {f}")

    cols = list(dict.fromkeys(["Qid", "audio", "Question", gold_col, *meta_cols]))
    rows = pq.read_table(f, columns=cols).to_pylist()
    rows.sort(key=lambda r: r["Qid"])  # stable enumeration order before sampling (contract step 2)

    rng = np.random.default_rng(seed)
    idx = _sample_indices(rng, len(rows), n)

    cache_dir = wav_cache_dir("vocalbench")
    out, ids = [], []
    for j in idx:
        r = rows[int(j)]
        item_id = r["Qid"]
        wav = wav_from_bytes(r["audio"]["bytes"], f"{axis}_{item_id}", cache_dir)
        meta = {"dataset": dataset, "item_id": item_id, "split": split, "task": task,
                "question": r.get("Question")}
        for c in meta_cols:
            if c == gold_col:
                continue
            v = r.get(c)
            if c == "Context":
                meta["context"] = _flatten_context(v)
            else:
                meta[c.lower()] = v
        out.append({"wav": wav, "gold": r[gold_col], "meta": meta})
        ids.append(item_id)

    freeze_ids(dataset, ids, dataset=dataset, seed=seed, extra={"split": split, "n_requested": n})
    return out


def load_vocalbench_knowledge(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VocalBench knowledge: 2000 short-fact spoken QA, free-text gold answer.

    Row: wav = materialized audio; gold = Answer (str);
    meta = {..., "task": "qa", "question": Question, "topic": Topic, "source": Source}.
    """
    return _load_axis("knowledge", gold_col="Answer", task="qa", meta_cols=("Topic", "Source"),
                       split=split, n=n, seed=seed)


def load_vocalbench_reasoning(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VocalBench reasoning: 1000 short spoken reasoning QA, free-text gold answer.

    Row: wav = materialized audio; gold = Answer (str);
    meta = {..., "task": "qa", "question": Question, "category": Category, "source": Source}.
    """
    return _load_axis("reasoning", gold_col="Answer", task="qa", meta_cols=("Category", "Source"),
                       split=split, n=n, seed=seed)


def load_vocalbench_multi_round(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VocalBench multi_round: 400 rows, prior text Context + ONE new spoken follow-up question.

    ``audio`` renders the NEW follow-up ``Question`` only (not the whole conversation) -- the
    prior turns are flattened into ``meta["context"]`` as inline text (see module docstring's
    ``_flatten_context``), so this is runnable single-turn: text context + audio question -> gold.

    Row: wav = materialized audio (the follow-up question); gold = Answer (str);
    meta = {..., "task": "qa", "question": Question, "context": <flattened Context text>,
    "category": Category}.
    """
    return _load_axis("multi_round", gold_col="Answer", task="qa", meta_cols=("Context", "Category"),
                       split=split, n=n, seed=seed)


def load_vocalbench_emotion(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VocalBench emotion: 500 rows, spoken emotionally-loaded question -> SER classification.

    gold = Question_emo, a 5-way closed label (verified 2026-07-09: angry/happy/neutral/sad/
    surprised, exactly 100 rows each -- a caller building a fixed K4 label set can hardcode this
    5-way set, or derive it from a sample per the ``uro-bench-UnderEmotion`` precedent in
    ``uro_bench.py`` / ``run_baseline._load_rows``). Note this axis has NO ``Answer`` column
    (unlike knowledge/reasoning/multi_round) -- only ``Score`` (list[int], 5 raw human intensity
    ratings for the emotion label itself), kept in meta for reference only, is available.

    Row: wav = materialized audio; gold = Question_emo (str label);
    meta = {..., "task": "emotion", "question": Question, "score": Score}.
    """
    return _load_axis("emotion", gold_col="Question_emo", task="emotion",
                       meta_cols=("Score",), split=split, n=n, seed=seed)


_ALL = {
    "vocalbench-knowledge": load_vocalbench_knowledge,
    "vocalbench-reasoning": load_vocalbench_reasoning,
    "vocalbench-multi-round": load_vocalbench_multi_round,
    "vocalbench-emotion": load_vocalbench_emotion,
}


if __name__ == "__main__":
    for name, fn in _ALL.items():
        try:
            rows = fn(n=3)
            keys = sorted(rows[0].keys()) if rows else []
            print(f"{name}: n={len(rows)} keys={keys} gold0={repr(rows[0]['gold'])[:80] if rows else None}")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
