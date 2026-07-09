"""cn-celeb1 — CN-Celeb (v1) speaker-ID loader (task B2-paralinguistic).

Layout on disk (SPEECHRL_DATA_DIR/datasets/cn-celeb1/CN-Celeb_flac/):
    data/id00000/{genre}-{session}-{utt}.flac    -- full 997-speaker corpus (not used here)
    eval/test/{spk}-{genre}-{session}-{utt}.flac -- 17777 test utterances, 200 held-out speakers
    eval/enroll/{spk}-enroll.flac                -- 196 per-speaker enrollment utterances
    eval/lists/{enroll.lst, enroll.map, test.lst, trials.lst}

**Discrepancy from the recipe, documented here**: this is a re-encoded HF mirror
("CN-Celeb_flac") -- confirmed empirically 2026-07-09: the official list files
(enroll.lst / test.lst / trials.lst) reference paths with a ``.wav`` extension (e.g.
``test/id00800-singing-01-001.wav``), but the audio actually materialized on disk under
``eval/`` has been re-encoded to ``.flac`` (same basename, different extension; verified via
directory listing). Both ``load_cn_celeb1`` and ``trial_pairs`` therefore swap the list
files' nominal ``.wav`` suffix for the real ``.flac`` path before returning it -- callers
never see the stale extension.

Filename code for eval/test (per CN-Celeb README.TXT + confirmed empirically): 4 '-'-joined
fields ``{speaker_id}-{genre}-{session_id}-{utter_id}``. Genre never itself contains a '-'
(multi-word genres use '_', e.g. "live_broadcast" for the README's "Live Broadcast"), so a
plain ``str.split("-")`` into exactly 4 parts is safe and was verified against all 11 genres
present in ``eval/test/`` (advertisement, drama, entertainment, interview, live_broadcast,
movie, play, recitation, singing, speech, vlog).

Enrollment files (``eval/enroll/{spk}-enroll.flac``) are each a concatenation of several
source utterances (see ``enroll.map``) spanning potentially different genres, so they have no
single well-defined ``genre`` -- ``gold["genre"]`` is ``None`` for ``split="enroll"`` rows.

Official trial pairs (verification protocol: same/different speaker) are exposed via a
SEPARATE accessor, ``trial_pairs()``, rather than folded into ``load_cn_celeb1`` -- a trial
is a PAIR of utterances + a same/different-speaker label, which doesn't fit the single-item
``Row`` contract, so it is intentionally not registered in ``registry.LOADERS``.
``trials.lst`` has ~3.48M lines (179MB); ``trial_pairs`` streams it in two passes (count,
then a single linear scan picking pre-selected line numbers) rather than loading it whole.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids  # noqa: E402


def _root() -> Path:
    return datasets_dir() / "cn-celeb1" / "CN-Celeb_flac"


def _parse_eval_test_stem(stem: str) -> dict:
    parts = stem.split("-")
    if len(parts) != 4:
        raise ValueError(f"cn-celeb1 eval/test filename does not match {{spk}}-{{genre}}-{{session}}-{{utt}}: {stem}")
    spk, genre, session, utt = parts
    return {"speaker_id": spk, "genre": genre, "session": session, "utt": utt}


def load_cn_celeb1(split: str = "eval", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load CN-Celeb1 rows from ``eval/test`` (split="eval", default) or ``eval/enroll``.

    gold = {"speaker_id", "genre"} for split="eval" (genre parsed from the filename); for
    split="enroll", ``gold["genre"]`` is None (see module docstring -- enrollment files span
    multiple genres). Use ``trial_pairs()`` for the official verification protocol.
    """
    import numpy as np

    root = _root()
    if split == "eval":
        audio_dir = root / "eval" / "test"
        flac_paths = sorted(audio_dir.glob("*.flac"))  # sorted() per README determinism rule
        if not flac_paths:
            raise FileNotFoundError(f"cn-celeb1 eval/test flac files not found under: {audio_dir}")
    elif split == "enroll":
        audio_dir = root / "eval" / "enroll"
        flac_paths = sorted(audio_dir.glob("*.flac"))
        if not flac_paths:
            raise FileNotFoundError(f"cn-celeb1 eval/enroll flac files not found under: {audio_dir}")
    else:
        raise NotImplementedError(
            f"cn-celeb1 split={split!r} not wired up -- only 'eval' (eval/test/) and 'enroll' "
            "(eval/enroll/) are implemented; the full 997-speaker data/ tree (dev pool) is out "
            "of scope for B2-paralinguistic."
        )

    rng = np.random.default_rng(seed)
    if n is not None and n < len(flac_paths):
        idx = rng.choice(len(flac_paths), size=n, replace=False)
        idx.sort()
        flac_paths = [flac_paths[int(i)] for i in idx]

    rows = []
    sampled_ids = []
    for p in flac_paths:
        item_id = f"cn-celeb1/{split}/{p.stem}"
        if split == "eval":
            parsed = _parse_eval_test_stem(p.stem)
            gold = {"speaker_id": parsed["speaker_id"], "genre": parsed["genre"]}
            meta = {"dataset": "cn-celeb1", "item_id": item_id, "split": split,
                     "session": parsed["session"], "utt": parsed["utt"]}
        else:  # enroll: "{spk}-enroll"
            spk = p.stem.split("-enroll")[0]
            gold = {"speaker_id": spk, "genre": None}
            meta = {"dataset": "cn-celeb1", "item_id": item_id, "split": split}
        rows.append({"wav": str(p), "gold": gold, "meta": meta})
        sampled_ids.append(item_id)

    freeze_ids("cn-celeb1", sampled_ids, dataset="cn-celeb1", seed=seed, extra={"split": split})
    return rows


def trial_pairs(n: int | None = None, seed: int = SLICE_SEED) -> list[dict]:
    """Official CN-Celeb1 speaker-verification trial pairs (a SEPARATE accessor; see module docstring).

    Each returned dict: {"enroll_id", "enroll_wav", "test_item_id", "test_wav", "label"}
    where label is 1 (same speaker) or 0 (different speaker), read straight from
    ``eval/lists/trials.lst`` (col1=enroll_id, col2=test rel-path, col3=label). ``.wav``
    paths in the list file are resolved to the real ``.flac`` files on disk (see module
    docstring's discrepancy note).

    ``n=None`` returns ALL ~3.48M trials (179MB list file; the caller's problem to filter
    further) -- pass an explicit ``n`` to draw a seeded subsample instead, using a two-pass
    streaming read (count, then linear scan) rather than loading the whole file into memory.
    """
    import numpy as np

    root = _root()
    trials_path = root / "eval" / "lists" / "trials.lst"
    enroll_dir = root / "eval" / "enroll"
    test_dir = root / "eval" / "test"
    if not trials_path.exists():
        raise FileNotFoundError(f"cn-celeb1 trials.lst not found: {trials_path}")

    def _resolve(enroll_id: str, test_rel: str) -> tuple[str, str]:
        enroll_wav = str(enroll_dir / f"{enroll_id}.flac")
        test_name = Path(test_rel).name  # "id00800-singing-01-001.wav" -> swap ext below
        test_wav = str(test_dir / (Path(test_name).stem + ".flac"))
        return enroll_wav, test_wav

    if n is None:
        pairs = []
        with open(trials_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 3:
                    continue
                enroll_id, test_rel, label = parts
                enroll_wav, test_wav = _resolve(enroll_id, test_rel)
                test_item_id = f"cn-celeb1/eval/{Path(test_rel).stem}"
                pairs.append({"enroll_id": enroll_id, "enroll_wav": enroll_wav,
                               "test_item_id": test_item_id, "test_wav": test_wav,
                               "label": int(label)})
        freeze_ids("cn-celeb1-trials", [f"{p['enroll_id']}|{p['test_item_id']}" for p in pairs],
                   dataset="cn-celeb1", seed=seed, extra={"n_trials": len(pairs), "subsampled": False})
        return pairs

    # two-pass streaming sample: pass 1 counts lines, pass 2 picks pre-selected line numbers
    with open(trials_path, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    if n >= total:
        return trial_pairs(n=None, seed=seed)

    rng = np.random.default_rng(seed)
    target_lines = set(int(i) for i in rng.choice(total, size=n, replace=False))

    pairs = []
    with open(trials_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in target_lines:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            enroll_id, test_rel, label = parts
            enroll_wav, test_wav = _resolve(enroll_id, test_rel)
            test_item_id = f"cn-celeb1/eval/{Path(test_rel).stem}"
            pairs.append({"enroll_id": enroll_id, "enroll_wav": enroll_wav,
                           "test_item_id": test_item_id, "test_wav": test_wav,
                           "label": int(label)})
    pairs.sort(key=lambda p: (p["enroll_id"], p["test_item_id"]))  # stable output order

    freeze_ids("cn-celeb1-trials", [f"{p['enroll_id']}|{p['test_item_id']}" for p in pairs],
               dataset="cn-celeb1", seed=seed, extra={"n_trials": len(pairs), "subsampled": True})
    return pairs
