"""fleurs_r — FLEURS-R multilingual LID(+gender)+ASR loader (K3, coverage taxonomy).

Recipe: ``wiki/2026-07-09-coverage-dataset-taxonomy.md`` K3 ("fleurs-r: 12-lang balanced slice; LID
12-way EM + per-language WER + gender; CLAP multilingual <3% negative control") +
``survey/2026-07-09-datasets-lock-first14.md`` finding #3: on this box FLEURS-R has only 12
languages (af_za/am_et/ar_eg/as_in/ast_es/az_az/be_by/bg_bg/bn_in/bs_ba/ca_es/ceb_ph) — no en/zh, no
translation target — so the earlier "zh/en ST slice" plan does not hold; it is re-judged as a
multilingual LID + per-language ASR + gender task, which is what this loader implements.

**Extraction status — discrepancy vs the task text** ("after A2"): verified 2026-07-09, audio is
ALREADY extracted for all 12 languages under ``datasets/fleurs-r/data/<lang>/audio/<split>/*.wav``
(alongside the untouched ``.tar.gz`` archives) — no extraction step was needed; this loader is NOT
blocked.

Per-language ``<split>.tsv`` is the raw (unpacked) FLEURS 7-column format, tab-separated, NO header:
``id, filename, raw_transcription, transcription, phonemic_transcription, num_samples, gender``.
``transcription`` (lower-cased, light punctuation-normalized) is used as the WER gold; the untouched
``raw_transcription`` is kept in ``meta``. Note the tsv row count and the extracted-audio file count
are NOT identical for a given split (e.g. af_za/test.tsv has 264 rows but audio/test/ has 414 files)
— this loader is defensive: it only yields rows whose referenced wav actually exists on disk, and
records how many tsv rows were dropped for that reason.

Sampling is BALANCED across all 12 languages (per the taxonomy doc's "12 语平衡" instruction): the
requested ``n`` is split into per-language quotas (``n // 12``, remainder to the first languages in
sorted order), and rows are drawn via one shared ``numpy.random.default_rng(seed)`` consumed
sequentially over the sorted language list (deterministic given the fixed on-disk data).
``n=None`` returns every available row for every language.

``split`` accepts ``"test"`` (default), ``"dev"``, or ``"train"`` — mapping directly to the on-disk
``<lang>/<split>.tsv`` + ``<lang>/audio/<split>/`` pair.

Row shape:
    wav  -> path to the on-disk wav (no materialization needed; already real per-split wav files)
    gold -> {"lang": <lang dir name, e.g. "af_za">, "transcript": <transcription>, "gender": <str>}
    meta -> {"dataset": "fleurs-r", "item_id": "<lang>:<tsv id>", "split": <split>,
             "raw_transcription": <str>, "num_samples": <int>}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids  # noqa: E402

_REVISION = None  # see docs/datasets.lock.json
_SPLITS = ("test", "dev", "train")
_LANGS = ("af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by", "bg_bg", "bn_in", "bs_ba",
          "ca_es", "ceb_ph")


def _read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 7:
                continue  # malformed row -- skip rather than crash the whole slice
            fid, filename, raw_transcription, transcription, phonemic, num_samples, gender = parts
            rows.append({
                "id": fid,
                "filename": filename,
                "raw_transcription": raw_transcription,
                "transcription": transcription,
                "num_samples": num_samples,
                "gender": gender,
            })
    return rows


def load_fleurs_r(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    import numpy as np

    if split not in _SPLITS:
        raise ValueError(f"fleurs-r: unknown split {split!r} (accepts: {_SPLITS})")

    fl_dir = datasets_dir() / "fleurs-r" / "data"
    per_lang = {}
    for lang in _LANGS:
        tsv = fl_dir / lang / f"{split}.tsv"
        audio_dir = fl_dir / lang / "audio" / split
        if not tsv.exists():
            raise FileNotFoundError(f"fleurs-r: missing {tsv}")
        rows = _read_tsv(tsv)
        rows.sort(key=lambda r: r["id"])
        avail = []
        n_dropped = 0
        for r in rows:
            wp = audio_dir / r["filename"]
            if wp.exists():
                r["_wav"] = str(wp)
                avail.append(r)
            else:
                n_dropped += 1
        per_lang[lang] = (avail, n_dropped)

    rng = np.random.default_rng(seed)
    if n is None:
        quotas = {lang: len(per_lang[lang][0]) for lang in _LANGS}
    else:
        base = n // len(_LANGS)
        rem = n % len(_LANGS)
        quotas = {lang: base + (1 if i < rem else 0) for i, lang in enumerate(_LANGS)}

    out = []
    ids = []
    for lang in _LANGS:
        avail, n_dropped = per_lang[lang]
        q = min(quotas[lang], len(avail))
        idx = rng.permutation(len(avail))[:q] if avail else np.array([], dtype=int)
        for j in idx:
            r = avail[int(j)]
            item_id = f"{lang}:{r['id']}"
            ids.append(item_id)
            out.append({
                "wav": r["_wav"],
                "gold": {"lang": lang, "transcript": r["transcription"], "gender": r["gender"]},
                "meta": {
                    "dataset": "fleurs-r",
                    "item_id": item_id,
                    "split": split,
                    "raw_transcription": r["raw_transcription"],
                    "num_samples": r["num_samples"],
                },
            })

    freeze_ids("fleurs-r", ids, dataset="fleurs-r", revision=_REVISION, seed=seed,
               extra={"split": split, "n_requested": n, "langs": list(_LANGS),
                      "n_dropped_per_lang": {lang: per_lang[lang][1] for lang in _LANGS}})
    return out


if __name__ == "__main__":
    from _common import data_root

    os.environ.setdefault("SPEECHRL_DATA_DIR", str(data_root()))
    rows = load_fleurs_r(n=6)
    for r in rows:
        print(r["meta"]["item_id"], "gold=", r["gold"], "wav_exists=", os.path.exists(r["wav"]))
