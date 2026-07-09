"""voicebench — VoiceBench (ModelScope lmms-lab mirror) re-examined include-subset loaders
(task B5-reexamined).

Recipe: ``wiki/survey/2026-07-09-datasets-lock-second14.md``'s "voicebench" section -- re-examined
from a full exclude to an **include-subset** verdict: 5 verifiable subsets (openbookqa, mmsu-spoken,
sd-qa, bbh, ifeval) + 1 prompt-only refusal probe (advbench). Everything else in the upstream repo
(alpacaeval / alpacaeval_full / alpacaeval_speaker / commoneval / wildvoice / mtbench) stays
EXCLUDED -- those are open-ended / judge-scored, no discrete reference, out of scope for a
training-free-verifiable-reward pipeline.

On-disk layout: ``datasets/voicebench/<subset>/*.parquet``. Every subset shares an ``audio``
column (``{"bytes": ..., "path": ... | None}``) -- the SPOKEN rendering of a text prompt -- plus
subset-specific text columns (see each loader's docstring). ``prompt`` (where present) is the
transcript/source-text the audio itself renders (TTS input), not extra ground-truth beyond what a
model could recover from the audio; loaders pass it through in ``meta`` for provenance / optional
reward-shaping. **Caller discipline note**: feeding ``meta["prompt"]`` straight into a model
instead of the audio would defeat the point of a speech-MLLM eval (info-boundary over-reach) --
that is a caller concern, not this loader's; the loader's only job is faithful data plumbing.

mmsu-spoken vs. mmsu (IMPORTANT, do not conflate): this is VoiceBench's own MMLU-derived spoken
benchmark (``datasets/voicebench/mmsu/<domain>-*.parquet``, 12 domains, letter-MCQ). It is a
DIFFERENT dataset from the standalone ``datasets/mmsu`` (MMSU audio-reasoning MCQ, 47 task_name
values, registered as ``"mmsu"`` in ``scripts/loaders/mmsu.py``) -- same short name upstream,
unrelated content. Registered here as ``"voicebench-mmsu-spoken"`` to keep the two apart.

Shared conventions across every subset loader in this module:
  - ``split`` accepted for interface-contract uniformity; only ``"test"`` is wired up (every
    VoiceBench subset on disk ships exactly one ``test-*.parquet`` family, no train/dev split) --
    anything else raises ``NotImplementedError``.
  - Multi-file subsets (mmsu-spoken: 12 per-domain files; sd-qa: 11 per-dialect files; bbh: 23
    shards) are glob'd with ``sorted()`` and concatenated in that order BEFORE sampling (README
    determinism rule) -- the per-file partition (domain / dialect) is preserved per-row in
    ``meta`` rather than discarded.
  - Rows with no natural id column (openbookqa, sd-qa, advbench) get a positional id
    ``f"{subset}#{i}"`` over the deterministic sorted-file/natural-row-order enumeration --
    stable as long as the pinned parquet content does not change (no revision is pinned by this
    loader; VoiceBench ships as a single unversioned ModelScope mirror snapshot on this box).
  - Every loader ends with ``freeze_ids`` so its exact sampled id set is replayable
    (README "Slice determinism" hard rule, step 3).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

# Matches a lettered MCQ option line, e.g. "A. a marsh" -- VoiceBench embeds the full spoken
# prompt (question + lettered options + a trailing "select one of A/B/C/D" instruction) as ONE
# text column; this is a best-effort convenience parse into meta["opts"], never the gold source
# (gold always comes from the dedicated `reference` column).
_MCQ_OPT_RE = re.compile(r"^([A-D])\.\s(.+)$", re.MULTILINE)


def _parse_mcq_prompt(prompt: str) -> tuple[str | None, list[str] | None]:
    """Best-effort (question, [options]) extraction from a VoiceBench MCQ `prompt` string."""
    pairs = _MCQ_OPT_RE.findall(prompt)
    if not pairs:
        return None, None
    question = prompt.split("\n", 1)[0].strip()
    return question, [text.strip() for _, text in pairs]


def _require_test_split(dataset: str, split: str) -> None:
    if split != "test":
        raise NotImplementedError(
            f"{dataset}: split={split!r} not wired up -- VoiceBench subsets ship a single "
            "test-*.parquet family only."
        )


def _sample_indices(rng, n_total: int, n: int | None):
    import numpy as np

    idx = rng.permutation(n_total) if n is not None else np.arange(n_total)
    return idx[:n] if n is not None else idx


# ---- openbookqa: 455-row single-file letter MCQ -------------------------------------------
def load_voicebench_openbookqa(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VoiceBench openbookqa: letter-MCQ (A-D), 1 file, 455 rows.

    Row: wav = materialized audio; gold = reference (single letter, e.g. "A");
    meta = {"dataset", "item_id", "split", "task": "mcq", "prompt", ["question", "opts"]}.
    """
    import numpy as np
    import pyarrow.parquet as pq

    _require_test_split("voicebench-openbookqa", split)
    f = next((datasets_dir() / "voicebench" / "openbookqa").glob("*.parquet"))
    rows = pq.read_table(f, columns=["audio", "prompt", "reference"]).to_pylist()

    rng = np.random.default_rng(seed)
    idx = _sample_indices(rng, len(rows), n)

    cache_dir = wav_cache_dir("voicebench")
    out, ids = [], []
    for i in idx:
        r = rows[int(i)]
        item_id = f"openbookqa#{int(i)}"
        question, opts = _parse_mcq_prompt(r["prompt"])
        meta = {"dataset": "voicebench-openbookqa", "item_id": item_id, "split": split,
                "task": "mcq", "prompt": r["prompt"]}
        if opts:
            meta["question"], meta["opts"] = question, opts
        out.append({"wav": wav_from_bytes(r["audio"]["bytes"], f"openbookqa_{int(i)}", cache_dir),
                    "gold": r["reference"], "meta": meta})
        ids.append(item_id)

    freeze_ids("voicebench-openbookqa", ids, dataset="voicebench/openbookqa", seed=seed,
               extra={"split": split, "n_requested": n})
    return out


# ---- mmsu-spoken: 12 per-domain files, letter MCQ (VoiceBench's spoken MMLU subset) --------
def load_voicebench_mmsu_spoken(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VoiceBench mmsu (spoken MMLU): letter-MCQ, 12 domain files merged into one pool.

    NOT the standalone ``mmsu`` dataset (see module docstring) -- registered as
    "voicebench-mmsu-spoken" to keep the two apart.

    Row: wav = materialized audio; gold = reference (single letter);
    meta = {..., "task": "mcq", "domain" (the parquet's own "category" column), "prompt",
    ["question", "opts"], "src" (upstream MMLU source tag)}.
    """
    import numpy as np
    import pyarrow.parquet as pq

    _require_test_split("voicebench-mmsu-spoken", split)
    files = sorted((datasets_dir() / "voicebench" / "mmsu").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("voicebench/mmsu: no domain parquet files found")

    cols = ["audio", "prompt", "reference", "question_id", "category", "src"]
    rows = []
    for f in files:
        rows += pq.read_table(f, columns=cols).to_pylist()

    rng = np.random.default_rng(seed)
    idx = _sample_indices(rng, len(rows), n)

    cache_dir = wav_cache_dir("voicebench")
    out, ids = [], []
    for i in idx:
        r = rows[int(i)]
        item_id = f"mmsu-spoken/{r['category']}/{r['question_id']}"
        question, opts = _parse_mcq_prompt(r["prompt"])
        meta = {"dataset": "voicebench-mmsu-spoken", "item_id": item_id, "split": split,
                "task": "mcq", "domain": r["category"], "src": r["src"], "prompt": r["prompt"]}
        if opts:
            meta["question"], meta["opts"] = question, opts
        out.append({"wav": wav_from_bytes(r["audio"]["bytes"], f"mmsuspoken_{int(i)}", cache_dir),
                    "gold": r["reference"], "meta": meta})
        ids.append(item_id)

    freeze_ids("voicebench-mmsu-spoken", ids, dataset="voicebench/mmsu", seed=seed,
               extra={"split": split, "n_requested": n})
    return out


# ---- sd-qa: 11 per-dialect files, short free-text answer (accent/ASR-robustness stratifier) -
def load_voicebench_sdqa(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VoiceBench sd-qa: short spoken-QA, 11 dialect files merged, dialect kept in meta.

    Row: wav = materialized audio; gold = reference (short free-text answer);
    meta = {..., "task": "qa-short", "dialect": <e.g. "usa","aus","ind_n",...>, "prompt"}.
    """
    import numpy as np
    import pyarrow.parquet as pq

    _require_test_split("voicebench-sd-qa", split)
    files = sorted((datasets_dir() / "voicebench" / "sd-qa").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("voicebench/sd-qa: no dialect parquet files found")

    dialect_re = re.compile(r"^(.+?)-\d+-of-\d+$")
    rows = []
    for f in files:
        m = dialect_re.match(f.stem)
        dialect = m.group(1) if m else f.stem
        for j, r in enumerate(pq.read_table(f, columns=["audio", "prompt", "reference"]).to_pylist()):
            rows.append((dialect, j, r))

    rng = np.random.default_rng(seed)
    idx = _sample_indices(rng, len(rows), n)

    cache_dir = wav_cache_dir("voicebench")
    out, ids = [], []
    for i in idx:
        dialect, j, r = rows[int(i)]
        item_id = f"sd-qa/{dialect}#{j}"
        meta = {"dataset": "voicebench-sd-qa", "item_id": item_id, "split": split,
                "task": "qa-short", "dialect": dialect, "prompt": r["prompt"]}
        out.append({"wav": wav_from_bytes(r["audio"]["bytes"], f"sdqa_{dialect}_{j}", cache_dir),
                    "gold": r["reference"], "meta": meta})
        ids.append(item_id)

    freeze_ids("voicebench-sd-qa", ids, dataset="voicebench/sd-qa", seed=seed,
               extra={"split": split, "n_requested": n})
    return out


# ---- bbh: 23 shards, Yes/No spoken reasoning -----------------------------------------------
def load_voicebench_bbh(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VoiceBench bbh (spoken BIG-Bench-Hard subset): binary Yes/No, 23 shards merged.

    Row: wav = materialized audio; gold = reference ("Yes" or "No");
    meta = {..., "task": "yesno", "prompt"} -- ``item_id`` reuses the parquet's own ``id`` column
    (globally unique, e.g. "bbh_navigate_147").
    """
    import numpy as np
    import pyarrow.parquet as pq

    _require_test_split("voicebench-bbh", split)
    files = sorted((datasets_dir() / "voicebench" / "bbh").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("voicebench/bbh: no shard parquet files found")

    rows = []
    for f in files:
        rows += pq.read_table(f, columns=["audio", "prompt", "reference", "id"]).to_pylist()
    rows.sort(key=lambda r: r["id"])  # stable enumeration order before sampling (contract step 2)

    rng = np.random.default_rng(seed)
    idx = _sample_indices(rng, len(rows), n)

    cache_dir = wav_cache_dir("voicebench")
    out, ids = [], []
    for i in idx:
        r = rows[int(i)]
        item_id = r["id"]
        meta = {"dataset": "voicebench-bbh", "item_id": item_id, "split": split,
                "task": "yesno", "prompt": r["prompt"]}
        out.append({"wav": wav_from_bytes(r["audio"]["bytes"], f"bbh_{item_id}", cache_dir),
                    "gold": r["reference"], "meta": meta})
        ids.append(item_id)

    freeze_ids("voicebench-bbh", ids, dataset="voicebench/bbh", seed=seed,
               extra={"split": split, "n_requested": n})
    return out


# ---- advbench: prompt-only refusal probe (NO reference column -- gold is always None) ------
def load_voicebench_advbench(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VoiceBench advbench: adversarial-prompt refusal probe, 1 file, 520 rows.

    There is no ``reference`` column upstream -- this is a behavioral probe (does the model
    refuse the harmful spoken request?), not a reference-scored task, so ``gold`` is always
    ``None`` and the loader marks it with ``meta["probe"] = "refusal"`` per the recipe. Scoring
    (a refusal classifier / judge) is explicitly a caller concern, not this loader's.

    Row: wav = materialized audio; gold = None;
    meta = {..., "task": "refusal-probe", "probe": "refusal", "prompt"}.
    """
    import numpy as np
    import pyarrow.parquet as pq

    _require_test_split("voicebench-advbench", split)
    f = next((datasets_dir() / "voicebench" / "advbench").glob("*.parquet"))
    rows = pq.read_table(f, columns=["audio", "prompt"]).to_pylist()

    rng = np.random.default_rng(seed)
    idx = _sample_indices(rng, len(rows), n)

    cache_dir = wav_cache_dir("voicebench")
    out, ids = [], []
    for i in idx:
        r = rows[int(i)]
        item_id = f"advbench#{int(i)}"
        meta = {"dataset": "voicebench-advbench", "item_id": item_id, "split": split,
                "task": "refusal-probe", "probe": "refusal", "prompt": r["prompt"]}
        out.append({"wav": wav_from_bytes(r["audio"]["bytes"], f"advbench_{int(i)}", cache_dir),
                    "gold": None, "meta": meta})
        ids.append(item_id)

    freeze_ids("voicebench-advbench", ids, dataset="voicebench/advbench", seed=seed,
               extra={"split": split, "n_requested": n})
    return out


# ---- ifeval: instruction-following, RULE-checkable (no reference text, no judge needed) ----
def load_voicebench_ifeval(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """VoiceBench ifeval: spoken instruction-following, 1 file, 345 rows.

    IFEval's whole point is that correctness is a PROGRAMMATIC rule check (e.g.
    "no commas allowed", "response must be >= N words") over ``instruction_id_list`` +
    ``kwargs``, not a text-reference match -- so ``gold`` is None and the rule-checking
    inputs live in ``meta`` verbatim, per the recipe.

    Row: wav = materialized audio; gold = None;
    meta = {..., "task": "ifeval", "prompt", "instruction_id_list": [...], "kwargs": [...]}.

    **Checker NOT ported (documented blocker, 2026-07-09)**: the task asked to port a
    permissive-licensed IFEval rule-checker into ``scripts/loaders/_ifeval_check.py`` IF one is
    locatable offline. Searched and found NONE in this environment:
      - ``pip list`` in the ``speechrl`` venv has no ``lm-eval`` / ``lighteval`` / ``nltk`` /
        ``langdetect`` (the google-research IFEval checker's own deps) and no package named
        ``*ifeval*``.
      - ``speechrl-data/repos/`` (the offline reference-repo cache) holds AudioGenie-Reasoner,
        JitRL, mbr-for-asr, slue-toolkit, slurp, TPO, TTRL -- none is
        ``google-research/instruction_following_eval`` (the canonical Apache-2.0 source of
        ``instructions_registry.py`` + ``instructions.py``).
    TODO: fetch ``google-research/google-research`` (Apache-2.0), subtree
    ``instruction_following_eval/``, into ``speechrl-data/repos/`` and port
    ``instructions_registry.INSTRUCTION_DICT`` + each ``Instruction.check_following`` into
    ``scripts/loaders/_ifeval_check.py`` (module not yet created) once network/offline access
    to that repo is available. Until then this loader ships the raw rule-check INPUTS
    (``instruction_id_list``/``kwargs``) but no checker -- callers cannot score ifeval rows yet.
    """
    import numpy as np
    import pyarrow.parquet as pq

    _require_test_split("voicebench-ifeval", split)
    f = next((datasets_dir() / "voicebench" / "ifeval").glob("*.parquet"))
    rows = pq.read_table(f, columns=["audio", "key", "prompt", "instruction_id_list", "kwargs"]).to_pylist()
    rows.sort(key=lambda r: r["key"])  # stable enumeration order before sampling (contract step 2)

    rng = np.random.default_rng(seed)
    idx = _sample_indices(rng, len(rows), n)

    cache_dir = wav_cache_dir("voicebench")
    out, ids = [], []
    for i in idx:
        r = rows[int(i)]
        item_id = str(r["key"])
        meta = {"dataset": "voicebench-ifeval", "item_id": item_id, "split": split,
                "task": "ifeval", "prompt": r["prompt"],
                "instruction_id_list": list(r["instruction_id_list"]), "kwargs": list(r["kwargs"])}
        out.append({"wav": wav_from_bytes(r["audio"]["bytes"], f"ifeval_{item_id}", cache_dir),
                    "gold": None, "meta": meta})
        ids.append(item_id)

    freeze_ids("voicebench-ifeval", ids, dataset="voicebench/ifeval", seed=seed,
               extra={"split": split, "n_requested": n})
    return out


_ALL = {
    "voicebench-openbookqa": load_voicebench_openbookqa,
    "voicebench-mmsu-spoken": load_voicebench_mmsu_spoken,
    "voicebench-sd-qa": load_voicebench_sdqa,
    "voicebench-bbh": load_voicebench_bbh,
    "voicebench-advbench": load_voicebench_advbench,
    "voicebench-ifeval": load_voicebench_ifeval,
}


if __name__ == "__main__":
    for name, fn in _ALL.items():
        try:
            rows = fn(n=3)
            keys = sorted(rows[0].keys()) if rows else []
            print(f"{name}: n={len(rows)} keys={keys} gold0={repr(rows[0]['gold'])[:80] if rows else None}")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
