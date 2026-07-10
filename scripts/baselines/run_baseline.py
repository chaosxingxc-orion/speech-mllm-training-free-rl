"""scripts/baselines/run_baseline.py — Stage-1 Step-1 baseline grid runner (DRAFT, criteria-freeze).

Grid = every dataset key in ``templates.DATASET_KTYPE`` (58 active Step-1 entries: 51 from
``scripts/loaders/registry.py``'s 60-loader package, minus the 2 K5-scoped-out SID loaders
[``cn-celeb1``, ``voxceleb1-test-split``], plus 7 legacy ``scripts/p2_baselines.py``-native
loaders that predate that package) x 2 backbones x {dev, test}. Per (dataset, backbone,
split): frozen-id rows -> FIXED instruction (``templates.build_instruction``) -> one backbone
generation call -> K-type scorer (``metrics.score``) -> aggregate + paired-bootstrap CI -> result
JSON under ``_repro/baselines/`` (this repo's established convention, see ``_repro/README.md`` --
committed, append-only research records).

BACKBONE ROSTER (owner ruling 2026-07-09, 双底座定稿 -- "the two-backbone roster is final"):
``qwen3-omni-30b-gguf`` and ``meralion-2-gguf`` are the ONLY backbones for Stage-1 waves 1-3. The
``minicpm-o-4.5``/``moss-audio-8b``/``nemotron`` HF-transformers stubs that used to sit in
``BACKBONES`` below are commented out, not deleted -- see that block for the rationale kept inline.

STAGE-1 DIRECTIONAL, CRITERIA-FREEZE DRAFT (2026-07-09): template/metric/grid code is
code-complete and import-clean; real execution against the two frozen GGUF backbones started with
the wave-1 sweep (``scripts/baselines/run_wave1.sh`` + ``wave1_cells.py`` -- see those files for
the frozen wave-1 cell list, checkpointing, and GPU-session lifecycle). ``--dry-run`` here still
exercises the whole grid-construction + template path with ZERO data-file or model dependencies
(see ``_synthetic_row``) for ad hoc single-cell/full-grid invocations outside wave-1.

reproduce (once run for real):
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data \\
  SPEECHRL_KB_DIR=/mnt/e/speechrl-knowledge \\
    python scripts/baselines/run_baseline.py --dataset librispeech --backbone qwen3-omni-30b-gguf --split test
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "loaders"))

import metrics                        # noqa: E402
import templates                      # noqa: E402
from _common import SLICE_SEED, data_root, freeze_ids  # noqa: E402  (scripts/loaders/_common.py)

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_SESSION_SH = REPO_ROOT / "scripts" / "gpu_session.sh"
OUT_DIR = REPO_ROOT / "_repro" / "baselines"

# ---- Stage-1 dev/test slice discipline ------------------------------------------------------
# No DEV_SEED/TEST_SEED convention existed anywhere in the codebase before this script (checked:
# no "dev_n"/"DEV_SEED"/"TEST_SEED" hit in scripts/*.py or wiki/2026-07-09-*.md as of 2026-07-09).
# A single shared seed (_common.SLICE_SEED) used for BOTH dev and test with different `n` would
# make dev a PREFIX of test (numpy's rng.permutation(seed) is identical regardless of n, so
# idx[:40] overlaps idx[:60] entirely) -- not a real held-out split. TEST_SEED is therefore a
# documented, deterministic OFFSET from SLICE_SEED, established HERE for the freeze meeting to
# ratify (or replace) -- see FREEZE_SHEET.md "dev/test seed convention".
DEV_SEED = SLICE_SEED            # 20260705
TEST_SEED = SLICE_SEED + 1000    # 20261705 -- NEW convention, pending freeze sign-off
DEV_N = 40
TEST_N = 60
NBOOT = 10000

BACKBONES: dict[str, dict] = {
    "qwen3-omni-30b-gguf": {
        "kind": "llamacpp",
        "server_env": "LLAMA_SERVER", "default_url": "http://127.0.0.1:8091",
        "gpu_session_name": "baselines-qwen3-omni-30b",
        "model_tag": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28)",
        "note": "resident via `scripts/gpu_session.sh serve qwen3-omni-30b-gguf up` (p2_baselines.gen pattern).",
    },
    "meralion-2-gguf": {
        "kind": "llamacpp",
        "server_env": "MERALION_SERVER", "default_url": "http://127.0.0.1:8197",
        "gpu_session_name": "baselines-meralion-2",
        "model_tag": "MERaLiON-2-3B GGUF (llama.cpp, -ngl 99)",
        "strip_prefix": "<Speaker1>:",
        "note": ("resident via `scripts/gpu_session.sh serve meralion-2-gguf up` (own port 8197, "
                 "pidfile, logfile -- gpu_session.sh's `serve <model-key>` was parameterized for this "
                 "second GGUF backbone, closing the infra gap this note used to flag). MERaLiON-2 "
                 "emits a literal '<Speaker1>: ' turn-prefix on every reply (G1-verified quirk, not "
                 "part of the answer) -- stripped in `generate()` below via `strip_prefix` BEFORE the "
                 "text ever reaches `metrics.score`, so no scorer needs to know about it."),
    },
    # --- minicpm-o-4.5 / moss-audio-8b / nemotron: REMOVED from the active roster -------------
    # Owner ruling 2026-07-09 (双底座定稿, "the two-backbone roster is final"): Stage-1 waves 1-3
    # run ONLY qwen3-omni-30b-gguf + meralion-2-gguf (above). These three HF-transformers stubs
    # (interface settled, weight-loading never implemented -- see the old FREEZE_SHEET.md
    # "backbone readiness" note) are commented out rather than deleted so the prior scoping
    # discussion isn't lost if a future wave reopens the roster:
    #
    # "minicpm-o-4.5": {
    #     "kind": "hf", "model_dir_env": "MINICPM_O_MODEL_DIR",
    #     "model_tag": "MiniCPM-o-4.5 (HF transformers)",
    # },
    # "moss-audio-8b": {
    #     "kind": "hf", "model_dir_env": "MOSS_AUDIO_MODEL_DIR",
    #     "model_tag": "MOSS-Audio-8B (HF transformers)",
    # },
    # "nemotron": {
    #     "kind": "hf", "model_dir_env": "NEMOTRON_MODEL_DIR",
    #     "model_tag": "Nemotron (HF transformers) -- 'possibly' per task brief",
    #     "note": ("owner sign-off needed: which nemotron checkpoint (ASR-only vs an omni/audio-LM "
    #              "variant) this backbone slot refers to -- wired identically to the other HF stubs "
    #              "pending that decision. See FREEZE_SHEET.md."),
    # },
}

# Loader keys whose registry.LOADERS callable needs a non-default `split` to select what this
# grid calls "dev"/"test" is NOT true here -- but seed-tts-eval overloads `split` for LANGUAGE
# (see scripts/loaders/seed_tts_eval.py), so its two K-type grid entries need special handling.
_SEED_TTS_LANG = {"seed-tts-eval-en": "en", "seed-tts-eval-zh": "zh"}

# Most registry loaders' eval split is literally named "test"; librispeech's is "test.clean" (see
# its module docstring -- "test" alone is not a directory on disk). One override map, checked
# before the generic split="test" fallback in _load_rows, rather than hardcoding "test" blindly.
_DEFAULT_SPLIT_OVERRIDE = {
    "librispeech": "test.clean",
    # HeySQuAD ships train/validation only -- no public test golds (mirrors load_heysquad's
    # module docstring); both dev and test grid slices draw from "validation".
    "heysquad": "validation",
}


# ---------------------------------------------------------------------------------------------
# generation adapters
# ---------------------------------------------------------------------------------------------

def sampling_params_for(backbone: str) -> dict:
    """Every sampling param actually sent for `backbone`'s generation calls, explicit -- for
    RECORDING alongside results (run_one's "sampling_params" field), not for building the request
    itself (gen_llamacpp below builds that payload directly). Previously that result field only
    recorded temperature/max_tokens/seed, leaving top_p/top_k/repeat_penalty implied-but-unrecorded
    even though the request payload already sent them explicitly (see p2_baselines.sampling_
    defaults' own docstring on why THOSE were pinned) -- this closes that recording gap. Shared
    across both llama.cpp backbones (qwen3-omni-30b-gguf and meralion-2-gguf both go through
    gen_llamacpp with the same p2.sampling_defaults()), so one function, not a per-backbone copy."""
    cfg = BACKBONES[backbone]
    base = {"temperature": 0.0, "max_tokens": 200}
    if cfg["kind"] == "llamacpp":
        import p2_baselines as p2  # lazy: reads SPEECHRL_DATA_DIR at import time

        base.update(p2.sampling_defaults())  # top_p / top_k / repeat_penalty, pinned 2026-07-09
    return base


def gen_llamacpp(base_url: str, wav_path: str, instruction: str, seed: int, temp: float = 0.0,
                  maxtok: int = 200) -> str:
    """Reuses scripts/p2_baselines.py's `gen()` request shape + `sampling_defaults()` verbatim
    (explicit sampling params pinned there 2026-07-09, see its docstring) -- not reimplemented.
    Backbone-agnostic (qwen3-omni-30b-gguf and meralion-2-gguf both call this with their own
    `base_url`); any backbone-specific post-processing (e.g. meralion's reply-prefix strip)
    happens in `generate()`, AFTER this returns, never here."""
    import p2_baselines as p2  # lazy: reads SPEECHRL_DATA_DIR at import time

    b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instruction}]}],
        "max_tokens": maxtok, "seed": seed, "temperature": temp, **p2.sampling_defaults()}
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


_HF_MODEL_CACHE: dict = {}


def _load_hf_backbone(backbone: str):
    """Lazy HF-checkpoint loader -- currently UNREACHABLE: the active BACKBONES roster (owner
    ruling 2026-07-09, 双底座定稿) has no `"kind": "hf"` entries left (see BACKBONES above, where
    the three HF stubs this fed are commented out, not deleted). Kept as settled scaffolding for
    if/when an HF backbone reopens, without pulling torch/transformers into `import run_baseline`
    (lazy-import discipline) in the meantime."""
    if backbone in _HF_MODEL_CACHE:
        return _HF_MODEL_CACHE[backbone]
    cfg = BACKBONES[backbone]
    raise NotImplementedError(
        f"HF backbone {backbone!r} is not wired up yet -- TODO: lazy-import torch/transformers "
        f"here, load the checkpoint from ${cfg['model_dir_env']}, cache it in _HF_MODEL_CACHE, "
        "and implement audio+text -> text generation matching gen_llamacpp's signature. "
        "See FREEZE_SHEET.md 'backbone readiness'."
    )


def gen_hf(backbone: str, wav_path: str, instruction: str, **_kw) -> str:
    _load_hf_backbone(backbone)  # always raises for now; see docstring above
    raise AssertionError("unreachable -- _load_hf_backbone always raises until implemented")


def generate(backbone: str, wav_path: str, instruction: str, seed: int) -> str:
    cfg = BACKBONES[backbone]
    if cfg["kind"] == "llamacpp":
        base_url = os.environ.get(cfg["server_env"], cfg["default_url"])
        text = gen_llamacpp(base_url, wav_path, instruction, seed)
        prefix = cfg.get("strip_prefix")
        if prefix and text.startswith(prefix):
            # G1-verified quirk: meralion-2-gguf emits a literal "<Speaker1>: " turn-prefix on
            # every reply that is never part of the actual answer -- stripped here, BEFORE
            # metrics.score ever sees the text, so no scorer needs a backbone-specific case.
            text = text[len(prefix):].lstrip()
        return text
    if cfg["kind"] == "hf":
        return gen_hf(backbone, wav_path, instruction)
    raise ValueError(f"unknown backbone kind {cfg['kind']!r} for {backbone!r}")


# ---------------------------------------------------------------------------------------------
# GPU session guard
# ---------------------------------------------------------------------------------------------

def assert_gpu_session_held(expected_owner: str) -> None:
    """Shell out to `scripts/gpu_session.sh check <expected_owner>` and require the lock to be
    HELD (recorded owner pid alive) by *someone*.

    Advisory, matches gpu_session.sh's own cooperative-lock discipline (see its module docstring)
    -- this only stops a well-behaved caller from generating without having acquired the shared
    GPU first, it cannot stop a rogue process.

    Does NOT hard-require the holder's name to equal `expected_owner` (`check` only warns on
    stderr if it differs, still exits 0). `expected_owner` (BACKBONES[...]["gpu_session_name"])
    documents the convention run_wave1.sh follows, but any legitimate caller -- a one-off manual
    invocation, an orchestrating agent using its own session identity, wave1's own
    "wave1-<backbone>" name -- may hold the lock under a different string and that is fine: what
    actually matters is that GPU access is arbitrated by ONE cooperative lock at all (CLAUDE.md GPU
    RULE), not the literal owner string.

    Uses `check` (not `status` + text-matching) so this Python code and the shell code share ONE
    definition of "held" -- see the 2026-07-09 postmortem in gpu_session.sh's header for why a
    naive pid-recorded-at-acquire-time check is easy to get wrong (transient-subshell pids).
    """
    if not GPU_SESSION_SH.exists():
        raise FileNotFoundError(f"gpu_session.sh not found at {GPU_SESSION_SH}")
    out = subprocess.run(["bash", str(GPU_SESSION_SH), "check", expected_owner],
                          capture_output=True, text=True, timeout=30)
    text = out.stdout + out.stderr
    if out.returncode != 0:
        raise RuntimeError(
            f"GPU session lock is not held -- acquire it first, e.g.:\n"
            f"  bash scripts/gpu_session.sh acquire {expected_owner} --pid \"$$\"\n"
            f"  bash scripts/gpu_session.sh serve <model-key> up\n"
            f"gpu_session.sh check said:\n{text.strip()}"
        )
    if "NOTE:" in text:
        print(text.strip(), file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# row loading (dispatches modern registry loaders, legacy p2_baselines loaders, and the
# synthetic K5-attribute / K7-slot / seed-tts-eval-<lang> sub-keys)
# ---------------------------------------------------------------------------------------------

def _legacy_rows(dataset_key: str, n: int, seed) -> list[dict]:
    """Adapt one scripts/p2_baselines.py LOADERS[...] callable's output into the Row shape
    ({"wav","gold","meta"}) this grid otherwise uses everywhere, WITHOUT reimplementing the
    loader itself. p2_baselines' own loaders bake `it["instr"]`/`it["task"]`/`it["opts"]` directly
    (predates the (split,n,seed)->Row contract) -- carried through verbatim in meta so
    templates.build_instruction's LEGACY_DATASETS passthrough and metrics.score's
    `p2.reward` reuse both see exactly what p2_baselines itself would have used."""
    import numpy as np
    import p2_baselines as p2  # lazy: reads SPEECHRL_DATA_DIR at import time

    p2.N_UTTS = n
    rng = np.random.default_rng(seed)
    items = p2.LOADERS[dataset_key](rng)
    rows, ids = [], []
    for i, it in enumerate(items):
        item_id = f"{dataset_key}#{i}"
        rows.append({
            "wav": it["wav"], "gold": it["gold"],
            "meta": {"dataset": dataset_key, "item_id": item_id, "split": "legacy",
                     "task": it["task"], "instr": it["instr"], "opts": it.get("opts")},
        })
        ids.append(item_id)
    freeze_ids(f"baselines-{dataset_key}", ids, dataset=dataset_key, seed=int(seed),
               extra={"adapter": "p2_baselines.LOADERS", "n_requested": n})
    return rows


def _observed_label_set(rows: list[dict], key_fn) -> list[str]:
    """Sample-observed closed label list (union of `key_fn(row)` over the CURRENT frozen sample).

    Documented limitation (see templates.py module docstring / FREEZE_SHEET.md): this is NOT the
    full corpus-level label inventory (e.g. SLURP's ~60 intents over 11,514 train + 2,974 test
    utterances) -- a rare label absent from this dev+test draw will not appear in the closed
    list shown to the model. Flagged as an open scoping question for the freeze meeting rather
    than silently under- or over-covering.
    """
    seen = []
    for r in rows:
        v = key_fn(r)
        if v and v not in seen:
            seen.append(v)
    return sorted(seen)


def _load_rows(dataset_key: str, split: str, n: int, seed: int) -> list[dict]:
    if dataset_key in templates.LEGACY_DATASETS:
        return _legacy_rows(dataset_key, n, seed)

    if dataset_key in _SEED_TTS_LANG:
        from seed_tts_eval import load_seed_tts_eval
        return load_seed_tts_eval(split=_SEED_TTS_LANG[dataset_key], n=n, seed=seed)

    import registry  # scripts/loaders/registry.py

    if dataset_key.endswith("-attr"):
        base_key = dataset_key[: -len("-attr")]
        attr = "speaker_sex"  # K5 Step-1 scope = speech-massive speaker_sex/speaker_age only
        rows = registry.LOADERS[base_key](split="validation", n=n, seed=seed)
        for r in rows:
            r["meta"]["_attr"] = attr
        return rows

    if dataset_key.endswith("-slot"):
        base_key = dataset_key[: -len("-slot")]
        rows = registry.LOADERS[base_key](split="test" if base_key == "slurp" else "validation", n=n, seed=seed)
        if base_key == "slurp":
            label_set = _observed_label_set(rows, lambda r: None)  # slurp slot types are per-entity;
            # populate from gold["entities"][*]["type"] across the sample instead of a scalar key:
            types = sorted({e["type"] for r in rows for e in (r["gold"].get("entities") or [])})
            for r in rows:
                r["meta"]["_label_set"] = types
        else:
            types = sorted({lab for r in rows for lab in (r["gold"].get("labels") or []) if lab and lab != "Other"})
            for r in rows:
                r["meta"]["_label_set"] = types
        return rows

    if dataset_key in ("slurp",):
        rows = registry.LOADERS[dataset_key](split="test", n=n, seed=seed)
        intents = _observed_label_set(rows, lambda r: r["gold"].get("intent"))
        for r in rows:
            r["meta"]["_label_set"] = intents
        return rows

    if dataset_key in ("speech-massive-de-DE", "speech-massive-fr-FR"):
        rows = registry.LOADERS[dataset_key](split="validation", n=n, seed=seed)
        intents = _observed_label_set(rows, lambda r: r["gold"].get("intent_str"))
        for r in rows:
            r["meta"]["_label_set"] = intents
        return rows

    if dataset_key in ("uro-bench-UnderEmotion-en", "uro-bench-UnderEmotion-zh"):
        rows = registry.LOADERS[dataset_key](split="test", n=n, seed=seed)
        emos = _observed_label_set(rows, lambda r: r.get("gold"))
        for r in rows:
            r["meta"]["_label_set"] = emos
        return rows

    split_arg = _DEFAULT_SPLIT_OVERRIDE.get(dataset_key, "test")
    return registry.LOADERS[dataset_key](split=split_arg, n=n, seed=seed)


# ---------------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------------

def paired_bootstrap(scores: list, nboot: int = NBOOT, seed: int = SLICE_SEED):
    """95% CI on the mean of ``scores`` (None entries dropped), 10k resamples by default.

    Mirrors ``scripts/p6_perception_delta.py:boot`` / ``scripts/repro_asr_best_of_n_llamacpp.py:agg``
    (the established paired-bootstrap pattern in this repo) -- reused, not reinvented.
    """
    import numpy as np

    arr = np.array([s for s in scores if s is not None], dtype=float)
    if len(arr) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(nboot)])
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


# ---------------------------------------------------------------------------------------------
# per-cell runner
# ---------------------------------------------------------------------------------------------

def run_one(dataset_key: str, backbone: str, split: str, n: int | None) -> dict:
    seed = DEV_SEED if split == "dev" else TEST_SEED
    n = n if n is not None else (DEV_N if split == "dev" else TEST_N)

    assert_gpu_session_held(BACKBONES[backbone].get("gpu_session_name", backbone))

    rows = _load_rows(dataset_key, split, n, seed)
    kt = templates.k_type_of(dataset_key) if dataset_key not in templates.LEGACY_DATASETS else "K8/K6 (legacy)"

    per_item, scores = [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        try:
            instr = templates.build_instruction(dataset_key, row)
            text = generate(backbone, row["wav"], instr, seed=seed + i)
            result = metrics.score(dataset_key, row, text)
        except Exception as e:  # noqa: BLE001 -- one bad item must not sink the whole cell
            per_item.append({"item_id": row.get("meta", {}).get("item_id"),
                              "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        per_item.append({"item_id": row["meta"]["item_id"], "instr": instr, "reply": text,
                          "score": result["score"], "detail": result["detail"]})
        scores.append(result["score"])
        if (i + 1) % 20 == 0:
            print(f"  [{dataset_key}/{backbone}/{split} {i+1}/{len(rows)}] "
                  f"({time.time()-t0:.0f}s)", flush=True)

    numeric = [s for s in scores if isinstance(s, (int, float))]
    mean = round(sum(numeric) / len(numeric), 4) if numeric else None
    ci = paired_bootstrap(numeric, seed=seed) if numeric else None

    aggregate = {"mean": mean, "ci95": ci, "n_scored": len(numeric), "n_total": len(rows)}
    if kt == "K4":
        pairs = [(it["detail"]["gold_label"], it["detail"]["pred_label"]) for it in per_item if "detail" in it and "gold_label" in it.get("detail", {})]
        if pairs:
            aggregate["macro_f1"] = metrics.aggregate_macro_f1(pairs)
    if kt == "K7":
        aggregate["note"] = "K7 per-item score is always None; run a separate aggregate_slot_f1_* pass over parsed_slots."

    snapshot_ref = f"SPEECHRL_KB_DIR/snapshots/baselines-{dataset_key}/sample_manifest.json" \
        if dataset_key in templates.LEGACY_DATASETS else \
        f"SPEECHRL_KB_DIR/snapshots/{dataset_key}/sample_manifest.json"

    out = {
        "stage": "1-directional",
        "dataset": dataset_key, "k_type": kt, "backbone": backbone, "split": split,
        "n_requested": n, "seed": seed,
        "sample_manifest": snapshot_ref,
        "model": BACKBONES[backbone]["model_tag"],
        "sampling_params": {"greedy_seed_base": seed, **sampling_params_for(backbone)},
        "boundary": ("audio-only input + fixed task-definition text (label set / MCQ options / "
                     "tool registry / JSON schema); no golden transcript/answer/intent ever placed "
                     "in the prompt -- see templates.py module docstring's Information-Boundary-Guard note."),
        "per_item": per_item,
        "aggregate": aggregate,
        "reproduce": (
            "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data "
            "SPEECHRL_KB_DIR=/mnt/e/speechrl-knowledge python scripts/baselines/run_baseline.py "
            f"--dataset {dataset_key} --backbone {backbone} --split {split} --n {n}"
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    return out


def write_result(result: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{result['dataset']}__{result['backbone']}__{result['split']}.json"
    json.dump(result, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return p


# ---------------------------------------------------------------------------------------------
# dry-run grid listing (zero data/model dependencies)
# ---------------------------------------------------------------------------------------------

def _synthetic_row(dataset_key: str) -> dict:
    """A minimal placeholder Row good enough for templates.build_instruction to render a preview
    WITHOUT touching SPEECHRL_DATA_DIR or any loader -- used only by --dry-run."""
    if dataset_key in templates.LEGACY_DATASETS:
        return {"wav": "(n/a)", "gold": "(n/a)",
                "meta": {"instr": "(passthrough of p2_baselines' own baked instruction -- not re-derived here)"}}
    kt = templates.k_type_of(dataset_key)
    meta = {"task": None, "opts": None}
    gold = None
    if kt == "K3":
        gold = {"lang": "af_za", "transcript": "(n/a)", "gender": "MALE"}
    elif kt == "K4":
        gold = {"emo": "neutral"}
    elif kt == "K5":
        gold = {"speaker_sex": "Male"}
        meta["_attr"] = "speaker_sex"
    elif kt == "K6":
        gold = {"intent": "(n/a)"}
    elif kt == "K8":
        gold = "(n/a)"
        meta["question"] = "(spoken question)"
    elif kt == "K10":
        meta["functions"] = [{"tool_name": "example_tool", "signature": "arg1: str"}]
    return {"wav": "(n/a)", "gold": gold, "meta": meta}


def dry_run(datasets: list[str], backbones: list[str], splits: list[str]) -> None:
    print(f"=== Stage-1 Step-1 baseline grid: {len(datasets)} datasets x {len(backbones)} backbones "
          f"x {len(splits)} splits = {len(datasets) * len(backbones) * len(splits)} cells ===\n")
    for dk in datasets:
        kt = "legacy" if dk in templates.LEGACY_DATASETS else templates.k_type_of(dk)
        try:
            instr = templates.build_instruction(dk, _synthetic_row(dk))
            preview = instr.replace("\n", " \\n ")[:100]
        except Exception as e:  # noqa: BLE001 -- dry-run must never crash on one bad template
            preview = f"<template error: {type(e).__name__}: {e}>"
        n_dev, n_test = DEV_N, TEST_N
        print(f"{dk:45s} [{kt:>4s}]  dev(n={n_dev}) test(n={n_test})  backbones={','.join(backbones)}")
        print(f"    template preview: {preview}...")
    print(f"\nexcluded from grid (K5 zero-shot-SID scoping, see templates.py): {sorted(templates.K5_EXCLUDED_SID)}")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def _all_dataset_keys() -> list[str]:
    return sorted(templates.DATASET_KTYPE.keys())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ALL", help="dataset key (see templates.DATASET_KTYPE) or ALL")
    ap.add_argument("--backbone", default="ALL", choices=[*BACKBONES.keys(), "ALL"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=None, help="override the frozen DEV_N/TEST_N for this run")
    ap.add_argument("--dry-run", action="store_true", help="list the grid + template previews; no model calls")
    args = ap.parse_args()

    datasets = _all_dataset_keys() if args.dataset == "ALL" else [args.dataset]
    backbones = list(BACKBONES.keys()) if args.backbone == "ALL" else [args.backbone]

    if args.dry_run:
        dry_run(datasets, backbones, [args.split])
        return

    for dk in datasets:
        for bb in backbones:
            print(f"=== running {dk} / {bb} / {args.split} ===", flush=True)
            try:
                result = run_one(dk, bb, args.split, args.n)
            except Exception as e:
                print(f"  !! {dk}/{bb}/{args.split} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            p = write_result(result)
            print(f"  wrote {p} (mean={result['aggregate']['mean']} ci={result['aggregate']['ci95']})",
                  flush=True)


if __name__ == "__main__":
    main()
