"""uro_bench — URO-Bench bucket-A verifiable QA/MCQ subsets (task B4-qa-mcq).

Recipe: ``survey/2026-07-09-datasets-lock-second14.md``'s "uro-bench 40 子集逐个开盘分桶" table
-- **bucket-A** (~15 subsets with a discrete/verifiable reference, included) vs **bucket-B**
(open-ended, judge-scored, excluded) vs **bucket-C** (needs the model to *produce* speech,
excluded). This module implements bucket-A only.

ONE shared parametrized loader (``_load_uro_subset``) backs all 15 subsets -- not 15 copies of
near-identical parsing code -- since every uro-bench parquet shares the same base schema
(``id``, ``source_wav``, ``source_text``, ``target_text``, [+dataset-specific extra cols]) and
differs only in *which* column is gold and whether the answer is an MCQ letter embedded in
``source_text``. ``SQuAD-zh`` and ``OpenbookQA-zh`` already had bespoke loaders in
``p2_baselines.py`` (``load_squad_zh``/``load_openbookqa_zh``, keyed by an rng-only signature
that predates this package's ``Row``/``(split, n, seed)`` contract) -- they are re-registered
here as thin shims over the same shared parametrized loader, not reimplemented from scratch, so
the bucket-A family is uniform end to end.

On-disk layout: ``datasets/uro-bench/<Subset>/test-*.parquet`` (one shard per subset, verified
2026-07-09). Common columns: ``id`` (int), ``source_wav`` (``{"bytes","path"}``, the spoken
question/prompt), ``source_text`` (the spoken prompt's transcript -- for MCQ subsets this
EMBEDS the answer options), ``target_text`` (the reference -- free text, a single MCQ letter,
or, for TruthfulEval, a ``list[str]`` of acceptable paraphrases). Extra per-subset columns
(``reference``, ``emotion``, ``language``, ``from_audio``) are read only where a subset needs
them.

Recipes (mirrors the survey table's "参考形式" column):
    extractive        SQuAD-zh                        gold = target_text (free-text span)
    mcq_letter         OpenbookQA-zh, GaokaoEval,       gold = target_text (single letter);
                        HSK5-zh                          options are parsed out of source_text
                                                          into meta["opts"] on a best-effort
                                                          basis (delimiters differ: "A. "/"A、"
                                                          for OpenbookQA-zh/GaokaoEval vs "A: "
                                                          for HSK5-zh -- one regex covers both)
    qa-numeric          Gsm8kEval                        gold = the dedicated ``reference`` col
                                                          (clean numeric string, e.g. "5") --
                                                          NOT target_text, which is CoT prose
    qa-numeric-zh        APE-zh                          gold = target_text (a full Chinese
                                                          sentence containing the numeric
                                                          answer, e.g. "五年级去了150人" -- this
                                                          loader does NOT extract the number;
                                                          that is reward-shaping, out of scope
                                                          for a thin loader)
    qa-short             MuChoEval-en                    gold = target_text (short phrase)
    qa-containment       MLC, MLC-zh, MLCpro-en,          gold = target_text (free-text/spoken-
                        MLCpro-zh                        arithmetic sentence; caller scores by
                                                          containment, not exact match)
    qa-echo              Repeat, Repeat-zh                gold = target_text (the clean string
                                                          to be echoed back, WITHOUT the
                                                          "please repeat after me" carrier
                                                          prefix that source_text has)
    emotion              UnderEmotion-en, UnderEmotion-zh  gold = the dedicated ``emotion`` col
                                                          (a classification label, not free
                                                          text) -- target_text (an empathetic
                                                          response, NOT the emotion word) is
                                                          kept in meta for reference only
    qa-truthful-weak     TruthfulEval                     gold = target_text, already a
                                                          ``list[str]`` of acceptable
                                                          paraphrases in the source parquet.
                                                          **Weak signal**: the real TruthfulQA
                                                          standard is a trained judge; list-
                                                          containment (any member appears in
                                                          the model's output) is a coarse
                                                          proxy that both under- and over-
                                                          credits paraphrases -- treat this
                                                          subset's numbers as directional only.

Row shape (uniform across all recipes):
    wav  -> materialized wav path (source_wav bytes decoded via _common.wav_from_bytes)
    gold -> recipe-dependent (see table above): str, or list[str] for TruthfulEval
    meta -> {"dataset": "uro-bench/<Subset>", "item_id": <id>, "split": <split>,
             "task": <recipe tag>, ["opts": [...] for mcq_letter when parseable],
             [subset-specific extra cols copied verbatim, e.g. "language", "emotion_reply"]}
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

# Matches an MCQ option label at the START of a segment: "A." / "A、" (OpenbookQA-zh, GaokaoEval)
# or "A:" (HSK5-zh). One regex covers every bucket-A mcq_letter subset seen on disk.
_OPT_SPLIT_RE = re.compile(r"(?=[A-D][.，、:])")
_OPT_PREFIX_RE = re.compile(r"^[A-D][.，、:]\s*")


def _parse_mcq_options(source_text: str) -> list[str] | None:
    """Best-effort extraction of embedded MCQ options from ``source_text``; None if unparseable.

    Metadata convenience only (matches p2_baselines.py's load_openbookqa_zh precedent) -- the
    loader NEVER falls back on this to compute gold; gold always comes straight from
    target_text.
    """
    parts = _OPT_SPLIT_RE.split(source_text)
    opts = []
    for p in parts:
        p = p.strip()
        if re.match(r"^[A-D][.，、:]", p):
            body = _OPT_PREFIX_RE.sub("", p).strip()
            body = re.sub(r"[;,，]\s*$", "", body).strip()
            if body:
                opts.append(body)
    return opts or None


def _load_uro_subset(
    subset: str,
    *,
    gold_col: str = "target_text",
    mcq: bool = False,
    task: str = "qa",
    meta_cols: tuple = (),
    split: str = "test",
    n: int | None = None,
    seed: int = SLICE_SEED,
) -> list:
    """Shared loader body for every uro-bench bucket-A subset -- see module docstring."""
    import numpy as np
    import pyarrow.parquet as pq

    if split != "test":
        raise NotImplementedError(f"uro-bench/{subset} split={split!r} not wired up -- only 'test' ships.")

    subset_dir = datasets_dir() / "uro-bench" / subset
    files = sorted(subset_dir.glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"uro-bench/{subset}: no test-*.parquet under {subset_dir}")

    cols = list(dict.fromkeys(["id", "source_wav", "source_text", gold_col, *meta_cols]))
    rows = []
    for f in files:
        rows += pq.read_table(f, columns=cols).to_pylist()
    rows.sort(key=lambda r: r["id"])  # stable enumeration order before sampling (contract step 2)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows)) if n is not None else np.arange(len(rows))
    idx = idx[:n] if n is not None else idx

    cache_dir = wav_cache_dir("uro-bench")
    slug = subset.lower().replace("-", "_")
    out, ids = [], []
    for j in idx:
        r = rows[int(j)]
        item_id = r["id"]
        wav = wav_from_bytes(r["source_wav"]["bytes"], f"{slug}_{item_id}", cache_dir)
        meta = {"dataset": f"uro-bench/{subset}", "item_id": item_id, "split": split, "task": task}
        if mcq:
            opts = _parse_mcq_options(r["source_text"])
            if opts:
                meta["opts"] = opts
        for c in meta_cols:
            if c != gold_col:
                meta[c] = r.get(c)
        out.append({"wav": wav, "gold": r[gold_col], "meta": meta})
        ids.append(item_id)

    freeze_ids(f"uro-bench-{slug}", ids, dataset=f"uro-bench/{subset}", seed=seed,
               extra={"split": split, "n_requested": n})
    return out


# ---- thin per-subset wrappers (the (split, n, seed) -> list[Row] contract registry.py wants) ----

def load_squad_zh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Shim over the shared loader -- p2_baselines.py's load_squad_zh predates this package."""
    return _load_uro_subset("SQuAD-zh", task="qa-extractive", split=split, n=n, seed=seed)


def load_openbookqa_zh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Shim over the shared loader -- p2_baselines.py's load_openbookqa_zh predates this package."""
    return _load_uro_subset("OpenbookQA-zh", mcq=True, task="mcq", split=split, n=n, seed=seed)


def load_gsm8k_eval(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("Gsm8kEval", gold_col="reference", task="qa-numeric",
                             split=split, n=n, seed=seed)


def load_gaokao_eval(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("GaokaoEval", mcq=True, task="mcq", split=split, n=n, seed=seed)


def load_hsk5_zh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("HSK5-zh", mcq=True, task="mcq", split=split, n=n, seed=seed)


def load_ape_zh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("APE-zh", task="qa-numeric-zh", split=split, n=n, seed=seed)


def load_muchoeval_en(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("MuChoEval-en", task="qa-short", meta_cols=("language", "from_audio"),
                             split=split, n=n, seed=seed)


def load_mlc(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("MLC", task="qa-containment", split=split, n=n, seed=seed)


def load_mlc_zh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("MLC-zh", task="qa-containment", split=split, n=n, seed=seed)


def load_mlcpro_en(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("MLCpro-en", task="qa-containment", meta_cols=("language",),
                             split=split, n=n, seed=seed)


def load_mlcpro_zh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("MLCpro-zh", task="qa-containment", meta_cols=("language",),
                             split=split, n=n, seed=seed)


def load_repeat(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("Repeat", task="qa-echo", split=split, n=n, seed=seed)


def load_repeat_zh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("Repeat-zh", task="qa-echo", split=split, n=n, seed=seed)


def load_underemotion_en(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("UnderEmotion-en", gold_col="emotion", task="emotion",
                             meta_cols=("language", "target_text"), split=split, n=n, seed=seed)


def load_underemotion_zh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return _load_uro_subset("UnderEmotion-zh", gold_col="emotion", task="emotion",
                             meta_cols=("language", "target_text"), split=split, n=n, seed=seed)


def load_truthfuleval(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """WEAK signal -- see module docstring's qa-truthful-weak recipe note."""
    return _load_uro_subset("TruthfulEval", task="qa-truthful-weak", split=split, n=n, seed=seed)


_ALL = {
    "SQuAD-zh": load_squad_zh,
    "OpenbookQA-zh": load_openbookqa_zh,
    "Gsm8kEval": load_gsm8k_eval,
    "GaokaoEval": load_gaokao_eval,
    "HSK5-zh": load_hsk5_zh,
    "APE-zh": load_ape_zh,
    "MuChoEval-en": load_muchoeval_en,
    "MLC": load_mlc,
    "MLC-zh": load_mlc_zh,
    "MLCpro-en": load_mlcpro_en,
    "MLCpro-zh": load_mlcpro_zh,
    "Repeat": load_repeat,
    "Repeat-zh": load_repeat_zh,
    "UnderEmotion-en": load_underemotion_en,
    "UnderEmotion-zh": load_underemotion_zh,
    "TruthfulEval": load_truthfuleval,
}


if __name__ == "__main__":
    for name, fn in _ALL.items():
        try:
            rows = fn(n=3)
            keys = sorted(rows[0].keys()) if rows else []
            print(f"{name}: n={len(rows)} keys={keys} gold0={repr(rows[0]['gold'])[:80] if rows else None}")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
