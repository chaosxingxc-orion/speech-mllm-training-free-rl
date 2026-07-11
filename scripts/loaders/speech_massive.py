"""speech_massive — Speech-MASSIVE multilingual SLU (intent + slot) loader (task B3-slu).

**License gate: CC-BY-NC-SA-4.0 -> internal Stage-1 evaluation ONLY** (survey
wiki/survey/2026-07-09-datasets-lock-first14.md, "speech-massive license" note). Never bundle
Speech-MASSIVE audio/labels with any released artifact.

Layout on disk: SPEECHRL_DATA_DIR/datasets/speech-massive/<locale>/{validation,train,
train_115}-*-of-*.parquet (13 locale dirs + a 14th ``all/`` dir that concatenates all locales'
``train`` shards only — ``all/`` is NOT exposed here, the recipe only calls for per-locale
splits). HF revision ``ff792febc16187a21e5bca38fb02a55daf91dc05`` (FBK-MT/Speech-MASSIVE, pinned
in docs/datasets.lock.json).

There is **no "test" split** for this dataset (discrepancy vs. the generic loader-interface
default): per the recipe, ``validation-*`` IS the eval split (e.g. fr-FR: 2 shards, 2034 rows);
``train-*`` is the memory/few-shot POOL (large, 9 shards for fr-FR/de-DE); ``train_115-*`` is a
small dedicated few-shot pool (exactly 115 rows, 1 shard). ``load_speech_massive``'s ``split``
default is therefore ``"validation"``, not ``"test"``.

Each parquet row already carries (id, locale, partition, scenario_str, intent_idx, intent_str,
utt, annot_utt, tokens, labels, slot_method, judgments, audio={"bytes","path"},
is_transcript_reported, is_validated, speaker_id, speaker_sex, speaker_age,
speaker_ethnicity_simple, speaker_country_of_birth/residence, speaker_nationality,
speaker_first_language) — verified against fr-FR's train_115 shard, 2026-07-09.

gold = {"intent_str": str, "annot_utt": str (slot-bracketed utterance, e.g.
        "allumer ma [device_type : prise]"), "tokens": [str, ...], "labels": [str, ...] (BIO-ish
        slot tags 1:1 with tokens, e.g. "Other"/"device_type"), "speaker_sex": str,
        "speaker_age": str} — the last two are bonus speaker-attribute gold (per the survey:
        "speaker_sex/age = 附赠说话人属性金标, 多语 SID 副测"), not requested by the primary SLU
        task but free in the source rows.

meta also carries "speaker_id" (2026-07-11, ticket #26 group-split design doc §1.3/§4.1) —
grouping/provenance only, never a task label: lets scripts/loaders/group_key.py split the K5
"-attr" (speaker-attribute) probe speaker-DISJOINT instead of falling back to item-level.

Efficiency note: each row embeds a full decoded wav (~150-400 KB) and shards run ~400 MB/9-shard
locale, so this loader reads parquet **row-group metadata only** (free: no data decode) to build
a global row index, then decodes just the row GROUPS (~100 rows each here) that intersect the
sampled slice -- never a whole shard file just to draw a small ``n``. Calling with ``n=None`` on
the ``train`` pool is still intentionally expensive (every row group of every shard IS the
request) -- call sparingly.

Generic core: ``load_speech_massive(locale, split, n, seed)`` works for any of the 12
(non-"all") locale directories (LOCALES below). Per task B3-slu, only the two smoke languages are
wired into the ``load_<name>``-shaped, registry-ready wrappers:
``load_speech_massive_fr_fr`` (fr-FR) and ``load_speech_massive_de_de`` (de-DE). Add another
``load_speech_massive_<xx_xx>`` one-line wrapper (+ a registry.py entry) to cover another
locale -- the loading logic itself needs no change.

Scoring is NOT implemented here (loader contract): intent = exact-match on gold["intent_str"];
slots = tag-level or span-level F1 over gold["labels"] vs. a caller's predicted tag sequence --
SPEECHRL_DATA_DIR/repos/slue-toolkit/slue_toolkit/eval/eval_utils_ner.py's ``get_ner_scores``
(precision/recall/F1 over (label, phrase, id) tuples) can score spans derived from
gold["tokens"]/gold["labels"] the same way it scores SLUE's own slot task.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

_REVISION = "ff792febc16187a21e5bca38fb02a55daf91dc05"  # FBK-MT/Speech-MASSIVE, docs/datasets.lock.json
LOCALES = ["ar-SA", "de-DE", "es-ES", "fr-FR", "hu-HU", "ko-KR", "nl-NL", "pl-PL", "pt-PT",
           "ru-RU", "tr-TR", "vi-VN"]  # the 12 per-locale dirs under datasets/speech-massive/ (excludes "all")
_SPLIT_PREFIX = {"validation": "validation", "train": "train", "train_115": "train_115"}
_COLUMNS = ["id", "locale", "partition", "scenario_str", "intent_idx", "intent_str", "utt",
            "annot_utt", "tokens", "labels", "audio", "speaker_sex", "speaker_age",
            # 2026-07-11 (ticket #26, group-split design doc §1.3/§4.1 K5 row): speaker_id was
            # exposed upstream but not read here -- a K5 speaker-ATTRIBUTE probe (speech-massive-
            # <locale>-attr) must split speaker-DISJOINT (the label IS speaker_sex/age, so any
            # same-speaker leakage across dev/test would let the model "cheat" via a different
            # utterance of the SAME speaker). Added to meta (not gold -- it's a grouping/provenance
            # id, not a task label) so scripts/loaders/group_key.py's "-attr" branch can resolve a
            # real speaker-disjoint group instead of falling back to item-level.
            "speaker_id"]


def _shard_files(locale: str, split: str) -> list:
    d = datasets_dir() / "speech-massive" / locale
    if not d.is_dir():
        raise FileNotFoundError(
            f"speech-massive: locale dir not found: {d} (available: {sorted(p.name for p in datasets_dir().glob('speech-massive/*') if p.is_dir())})"
        )
    prefix = _SPLIT_PREFIX.get(split)
    if prefix is None:
        raise ValueError(f"speech-massive: unknown split {split!r} (accepts: {sorted(_SPLIT_PREFIX)})")
    pat = re.compile(rf"^{re.escape(prefix)}-\d+-of-\d+\.parquet$")
    files = sorted(p for p in d.iterdir() if pat.match(p.name))  # sorted() per README's determinism rule
    if not files:
        raise FileNotFoundError(f"speech-massive: no {split!r} shards under {d}")
    return files


def load_speech_massive(locale: str, split: str = "validation", n: int | None = None,
                         seed: int = SLICE_SEED) -> list:
    """Load Speech-MASSIVE rows for (``locale``, ``split``) -- see module docstring for shapes.

    Determinism: a global row index is built from PER-SHARD, PER-ROW-GROUP metadata (file order
    sorted, row order within each row group is the file's own native order -- no re-sort of
    within-group rows since row groups are read/decoded as a unit); ``n`` then draws a seeded
    slice over that global index via ``numpy.random.default_rng(seed)``, matching the README
    contract. Only the row groups actually touched by the drawn slice are decoded.
    """
    import numpy as np
    import pyarrow.parquet as pq

    files = _shard_files(locale, split)
    pfs = [pq.ParquetFile(str(f)) for f in files]

    # global_index[i] = (file_idx, row_group_idx, local_row_idx) -- built from row-group METADATA
    # only (num_rows per group), no data decode, so this stays cheap even over the full `train` pool.
    global_index = []
    for fi, pf in enumerate(pfs):
        for rg in range(pf.metadata.num_row_groups):
            nrows = pf.metadata.row_group(rg).num_rows
            global_index.extend((fi, rg, li) for li in range(nrows))

    total = len(global_index)
    if n is not None and n < total:
        pick = sorted(int(i) for i in np.random.default_rng(seed).choice(total, size=n, replace=False))
    else:
        pick = list(range(total))

    # Decode each needed row group AT MOST ONCE (grouped by (file_idx, row_group_idx)).
    needed_groups = sorted({(global_index[gi][0], global_index[gi][1]) for gi in pick})
    tables = {(fi, rg): pfs[fi].read_row_group(rg, columns=_COLUMNS) for fi, rg in needed_groups}

    cache_dir = wav_cache_dir(f"speech-massive-{locale}")
    rows = []
    sampled_ids = []
    for gi in pick:
        fi, rg, li = global_index[gi]
        rec = tables[(fi, rg)].slice(li, 1).to_pylist()[0]
        # rec["id"] alone is not guaranteed globally unique across shards of a split -> disambiguate
        # with the (file, row_group, local) position, which is itself stable given a fixed sorted
        # shard file list (README determinism rule).
        item_id = f"{locale}/{split}/{rec['id']}@{fi}-{rg}-{li}"
        wav = wav_from_bytes(rec["audio"]["bytes"], f"{split}_{fi}_{rg}_{li}", cache_dir)
        rows.append({
            "wav": wav,
            "gold": {
                "intent_str": rec["intent_str"],
                "annot_utt": rec["annot_utt"],
                "tokens": rec["tokens"],
                "labels": rec["labels"],
                "speaker_sex": rec["speaker_sex"],
                "speaker_age": rec["speaker_age"],
            },
            "meta": {
                "dataset": f"speech-massive-{locale}",
                "item_id": item_id,
                "split": split,
                "scenario_str": rec["scenario_str"],
                "utt": rec["utt"],
                "speaker_id": rec["speaker_id"],  # 2026-07-11 ticket #26, see _COLUMNS comment above
            },
        })
        sampled_ids.append(item_id)

    freeze_ids(f"speech-massive-{locale}", sampled_ids, dataset=f"speech-massive-{locale}",
               revision=_REVISION, seed=seed, extra={"split": split, "n_requested": n})
    return rows


def load_speech_massive_fr_fr(split: str = "validation", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return load_speech_massive("fr-FR", split=split, n=n, seed=seed)


def load_speech_massive_de_de(split: str = "validation", n: int | None = None, seed: int = SLICE_SEED) -> list:
    return load_speech_massive("de-DE", split=split, n=n, seed=seed)


if __name__ == "__main__":
    from _common import data_root

    os.environ.setdefault("SPEECHRL_DATA_DIR", str(data_root()))
    for name, fn in (("fr-FR", load_speech_massive_fr_fr), ("de-DE", load_speech_massive_de_de)):
        print(f"--- {name} ---")
        rows = fn(n=3)
        for r in rows:
            print(r["meta"]["item_id"], "gold=", repr(r["gold"])[:160], "wav_exists=", os.path.exists(r["wav"]))
