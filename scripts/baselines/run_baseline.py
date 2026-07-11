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
import contextlib
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

import label_inventories               # noqa: E402  (scripts/baselines/label_inventories.py)
import metrics                        # noqa: E402
import provenance                     # noqa: E402  (scripts/baselines/provenance.py, ticket #25 P4)
import templates                      # noqa: E402
from _common import (                 # noqa: E402  (scripts/loaders/_common.py)
    POOL_RECONSTRUCTION_SEED, SLICE_SEED, data_root, freeze_ids, load_snapshot_ids,
)
from group_key import group_key_of    # noqa: E402  (scripts/loaders/group_key.py, ticket #26 --
# group-split design doc wiki/2026-07-11-group-split-statistics-design.md §2.5. Only used below to
# PERSIST group_id per per_item entry going forward -- nothing in this file reads it back yet
# (the group-disjoint redraw / cluster bootstrap wiring is a separate, owner-gated step).

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
        # 2026-07-10 wave-3 batched-inference: env vars gpu_session.sh's serve_up reads to build
        # THIS backbone's llama-server command line (see its LLAMA_NP/LLAMA_CACHE_RAM) -- read back
        # here (server_slot_config_for) so a result JSON's sampling_params reports the server config
        # actually used, not a guess. See that function's docstring.
        "np_env": "LLAMA_NP", "cache_ram_env": "LLAMA_CACHE_RAM",
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
        "np_env": "MERALION_NP", "cache_ram_env": "MERALION_CACHE_RAM",  # see qwen3 entry's comment above
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


def server_slot_config_for(backbone: str) -> dict:
    """The llama-server ``-np``/``--cache-ram`` values ACTUALLY used to start `backbone`'s resident
    server -- for RECORDING in a result JSON's ``sampling_params`` (2026-07-10 wave-3 batched-
    inference task brief item 1), never for building a request itself (that's per-item, unrelated to
    the server's own launch flags).

    Sourced from the SAME env vars ``gpu_session.sh``'s ``serve_up``/``_model_cfg`` read to build
    that backbone's llama-server command line (``BACKBONES[backbone]["np_env"]``/``"cache_ram_env"``)
    -- NOT queried over HTTP -- so this is guaranteed to reflect the exact flags the resident server
    was (or will be) launched with, not an independent guess that could drift from reality. Defaults
    ("-1"/"8192") match both llama-server's own compiled defaults AND gpu_session.sh's own defaults
    for these two env vars, so a caller that never touched them (every wave-1/wave-2 cell run before
    this task, all sequential / ``--parallel`` 1) is unaffected and reports the same values it always
    implicitly ran with.
    """
    cfg = BACKBONES[backbone]
    if cfg["kind"] != "llamacpp":
        return {"server_slots": None, "cache_ram_mib": None}
    return {"server_slots": int(os.environ.get(cfg["np_env"], "-1")),
            "cache_ram_mib": int(os.environ.get(cfg["cache_ram_env"], "8192"))}


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
                     "task": it["task"], "instr": it["instr"], "opts": it.get("opts"),
                     # 2026-07-11 (ticket #26, group-split design doc §1.3/§4.1): carries through
                     # p2_baselines' own per-item "group_key" (only mmau-mini populates it today,
                     # see that module's LOADERS comment for why the other 6 legacy keys don't) --
                     # None for every dataset that doesn't set it, which group_key.py treats as
                     # the honest G-NONE fallback, same as any other ungrouped item.
                     "group_key": it.get("group_key")},
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


def _maybe_restrict(rows: list[dict], restrict_ids: set | None) -> list[dict]:
    """Filter ``rows`` down to ``meta.item_id in restrict_ids`` (pool order preserved); no-op when
    ``restrict_ids`` is None. 2026-07-10 dev/test disjoint-redraw task: applied INSIDE ``_load_rows``
    (right after each branch's loader call, BEFORE that branch's label-set computation) rather than
    after ``_load_rows`` returns, so a ``--slice disjoint`` cell's sample-observed label sets
    (slurp/speech-massive intents, -slot types, UnderEmotion emotions -- see
    ``_observed_label_set``'s docstring) are computed over exactly the frozen slice's rows, same as
    a seeded cell computes them over its own drawn rows. Without this, the disjoint path's
    ``n=None`` full-pool fetch would silently swap every sample-observed label set for a full-pool
    one -- changing the PROMPT, when the redraw must change only the SLICE."""
    if restrict_ids is None:
        return rows
    return [r for r in rows if r["meta"]["item_id"] in restrict_ids]


def _load_rows(dataset_key: str, split: str, n: int, seed: int,
               restrict_ids: set | None = None) -> list[dict]:
    if dataset_key in templates.LEGACY_DATASETS:
        return _maybe_restrict(_legacy_rows(dataset_key, n, seed), restrict_ids)

    if dataset_key in _SEED_TTS_LANG:
        from seed_tts_eval import load_seed_tts_eval
        return _maybe_restrict(load_seed_tts_eval(split=_SEED_TTS_LANG[dataset_key], n=n, seed=seed),
                               restrict_ids)

    import registry  # scripts/loaders/registry.py

    if dataset_key.endswith("-attr"):
        base_key = dataset_key[: -len("-attr")]
        attr = "speaker_sex"  # K5 Step-1 scope = speech-massive speaker_sex/speaker_age only
        rows = _maybe_restrict(registry.LOADERS[base_key](split="validation", n=n, seed=seed), restrict_ids)
        for r in rows:
            r["meta"]["_attr"] = attr
        return rows

    if dataset_key.endswith("-slot"):
        base_key = dataset_key[: -len("-slot")]
        rows = _maybe_restrict(
            registry.LOADERS[base_key](split="test" if base_key == "slurp" else "validation", n=n, seed=seed),
            restrict_ids)
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
        rows = _maybe_restrict(registry.LOADERS[dataset_key](split="test", n=n, seed=seed), restrict_ids)
        intents = _observed_label_set(rows, lambda r: r["gold"].get("intent"))
        for r in rows:
            r["meta"]["_label_set"] = intents
        return rows

    if dataset_key in ("speech-massive-de-DE", "speech-massive-fr-FR"):
        rows = _maybe_restrict(registry.LOADERS[dataset_key](split="validation", n=n, seed=seed), restrict_ids)
        intents = _observed_label_set(rows, lambda r: r["gold"].get("intent_str"))
        for r in rows:
            r["meta"]["_label_set"] = intents
        return rows

    if dataset_key in ("uro-bench-UnderEmotion-en", "uro-bench-UnderEmotion-zh"):
        rows = _maybe_restrict(registry.LOADERS[dataset_key](split="test", n=n, seed=seed), restrict_ids)
        emos = _observed_label_set(rows, lambda r: r.get("gold"))
        for r in rows:
            r["meta"]["_label_set"] = emos
        return rows

    if dataset_key == "vocalbench-emotion":
        # 2026-07-10 freeze-repair (wave-2 audit): NO branch here ever populated
        # meta["_label_set"] for this dataset key before this fix -- it fell straight through to
        # the generic tail below with no per-dataset post-processing, so templates.build_instruction
        # rendered a ONE-OPTION prompt ("A. <observed set unavailable>") and metrics.score scored
        # against an EMPTY label set, making both wave-2 cells (dev/test) mechanically 0.0
        # regardless of the model's actual reply. Unlike the uro-bench-UnderEmotion branch just
        # above (sample-observed union from meta["_label_set"] -- later superseded by a corpus-true
        # templates.K4_LABEL_SETS entry), this uses the corpus-true, full-pool-verified 5-way set
        # DIRECTLY (label_inventories.VOCALBENCH_EMOTION_EMOTIONS -- 500/500 rows scanned, exactly
        # 100 each of angry/happy/neutral/sad/surprised, see that module's dedicated comment) since
        # the whole corpus is small enough that "sample-observed" and "corpus-true" coincide here;
        # no reason to route this through templates.K4_LABEL_SETS as well (that dict is reserved for
        # datasets whose label set is used regardless of meta, see templates.py K4 branch) -- see
        # label_inventories.py's module docstring for the full defect writeup.
        rows = _maybe_restrict(registry.LOADERS[dataset_key](split="test", n=n, seed=seed), restrict_ids)
        for r in rows:
            r["meta"]["_label_set"] = label_inventories.VOCALBENCH_EMOTION_EMOTIONS
        return rows

    split_arg = _DEFAULT_SPLIT_OVERRIDE.get(dataset_key, "test")
    return _maybe_restrict(registry.LOADERS[dataset_key](split=split_arg, n=n, seed=seed), restrict_ids)


# ---------------------------------------------------------------------------------------------
# disjoint-slice support (2026-07-10 dev/test disjoint-redraw task, owner ruling Decision-Log
# 续10) -- reads the FROZEN item-id manifests scripts/baselines/redraw.py writes, instead of
# re-drawing dev/test independently from DEV_SEED/TEST_SEED (the very thing that caused the
# overlap in the first place, see that module's docstring). Gated behind --slice disjoint /
# slice_mode="disjoint" below; --slice seeded (the default) is 100% unchanged prior behavior.
# ---------------------------------------------------------------------------------------------

def _disjoint_manifest_name(dataset_key: str, split: str) -> str:
    return f"baselines-{dataset_key}-disjoint-{split}"


@contextlib.contextmanager
def _suppress_loader_snapshot_writes():
    """No-op ``kb_snapshot.freeze_snapshot`` for the duration of the ``with`` block.

    2026-07-10 dev/test disjoint-redraw task: ``_load_rows_disjoint``'s full-pool refetch runs
    the REAL loaders (audio materialization included -- unlike ``redraw.py``'s metadata-only
    stub), and every loader (plus ``_legacy_rows`` above) unconditionally calls
    ``_common.freeze_ids`` under its OWN usual experiment name at the end of a load. Left alone,
    a ``--slice disjoint`` cell would therefore overwrite e.g. ``snapshots/aishell-1/`` (the OLD,
    task-brief-protected manifest name) with a full-pool 7176-id list as a side effect of merely
    READING the pool. Patching ``kb_snapshot.freeze_snapshot`` (one module attribute) suppresses
    every such write at the single choke point both ``freeze_ids`` and any direct caller resolve
    at call time -- the disjoint slice's OWN authoritative record is the
    ``baselines-<dataset>-disjoint-{dev,test}`` manifest redraw.py already froze, plus the result
    JSON's per_item list; there is nothing worth writing here.
    """
    here = os.path.dirname(os.path.abspath(__file__))                   # scripts/baselines
    knowledge_dir = os.path.join(os.path.dirname(here), "knowledge")    # scripts/knowledge
    if knowledge_dir not in sys.path:
        sys.path.insert(0, knowledge_dir)
    import kb_snapshot

    orig = kb_snapshot.freeze_snapshot
    kb_snapshot.freeze_snapshot = lambda *a, **kw: \
        "<suppressed during --slice disjoint pool refetch, see run_baseline._suppress_loader_snapshot_writes>"
    try:
        yield
    finally:
        kb_snapshot.freeze_snapshot = orig


def _load_rows_disjoint(dataset_key: str, split: str) -> list[dict]:
    """Load ``dataset_key``/``split`` rows from the FROZEN disjoint manifest rather than a fresh
    DEV_SEED/TEST_SEED draw. Re-fetches the SAME full natural pool ``redraw.py`` froze the
    manifest against (``_load_rows(dataset_key, split, n=None, seed=POOL_RECONSTRUCTION_SEED)`` --
    same seed constant, see its docstring on why that must match exactly for the 7 legacy
    p2_baselines datasets' fetch-position-based item ids), restricted to exactly the frozen
    item_ids (``restrict_ids`` is applied INSIDE ``_load_rows``, before label-set computation --
    see ``_maybe_restrict``'s docstring), then reordered to the manifest's own
    (already-disjoint-drawn) order.

    Raises if any frozen id is missing from the fresh pool refetch (pool drift since the manifest
    was frozen -- e.g. the underlying dataset file changed on disk) rather than silently returning
    a different/smaller slice than what was frozen.

    KNOWN COST (accepted, 2026-07-10): unlike redraw.py's metadata-only pool scan, this refetch
    runs the loaders for REAL, so bytes-embedding loaders (aishell-1, mmsu, spoken-squad, ...)
    materialize wav files for their ENTIRE pool on the first disjoint cell per dataset -- a
    one-time per-dataset cache fill (wav_from_bytes/_wav_from_bytes cache by key; the second
    split's cell and every later rerun/Phase-A pass hit the cache for free). Filtering before
    materialization would require editing every loader's internals; not worth it for a one-time
    cost.
    """
    manifest_name = _disjoint_manifest_name(dataset_key, split)
    frozen_ids = load_snapshot_ids(manifest_name)
    frozen_set = set(frozen_ids)
    with _suppress_loader_snapshot_writes():
        rows = _load_rows(dataset_key, split, n=None, seed=POOL_RECONSTRUCTION_SEED,
                          restrict_ids=frozen_set)
    by_id = {r["meta"]["item_id"]: r for r in rows}
    missing = [i for i in frozen_ids if i not in by_id]
    if missing:
        raise RuntimeError(
            f"{dataset_key}/{split}: {len(missing)}/{len(frozen_ids)} frozen id(s) from "
            f"{manifest_name!r} not found in a fresh full-pool refetch (first few: {missing[:5]}) "
            "-- pool drift since redraw.py froze this manifest?"
        )
    return [by_id[i] for i in frozen_ids]


# ---------------------------------------------------------------------------------------------
# locked-group-slice support (2026-07-11, ticket #26 task 5 -- PREP ONLY, see
# scripts/baselines/run_locked_rerun.sh's "DO NOT RUN" instruction: this code path is written and
# reviewable but not invoked by this ticket). Reads the group-aware LOCKED_HOLDOUT manifests
# ``scripts/baselines/locked_split.py`` writes -- the fresh, never-informed-a-decision
# ``LOCKED_TEST_SEED`` draw (design doc §2), NOT ``redraw.py``'s disjoint manifest (which reuses
# the already-decision-informing ``TEST_SEED`` and is therefore permanently non-locked, see
# ``_repro/redraw_manifest.NONLOCKED.md``). Gated behind --slice locked-group /
# slice_mode="locked-group" below; --slice seeded/disjoint (existing) are unchanged.
# ---------------------------------------------------------------------------------------------

def _locked_manifest_ids(dataset_key: str, split: str) -> list[str]:
    """Read ``_repro/LOCKED_HOLDOUT/<dataset_key>.json``'s ``test_ids``/``dev_ids`` via
    ``locked_split.load_locked_test``/``load_locked_dev``. Lazy, function-local import (NOT at
    module top) -- ``locked_split.py`` itself does ``import run_baseline as rb``, so importing it
    at THIS module's top level would be a circular import; deferring to call time (after both
    modules are already registered in ``sys.modules``) resolves the cycle, same lazy-import
    discipline this repo already applies to heavy deps."""
    import locked_split

    return locked_split.load_locked_test(dataset_key) if split == "test" else locked_split.load_locked_dev(dataset_key)


def _load_rows_locked(dataset_key: str, split: str) -> list[dict]:
    """Load ``dataset_key``/``split`` rows from the LOCKED_HOLDOUT manifest rather than a fresh
    DEV_SEED/TEST_SEED draw or ``redraw.py``'s (permanently non-locked) disjoint manifest. Mirrors
    ``_load_rows_disjoint``'s full-pool-refetch-then-restrict pattern exactly (same
    ``POOL_RECONSTRUCTION_SEED``, same ``_suppress_loader_snapshot_writes`` guard) -- the ids
    themselves come from ``locked_split.py``'s group-aware draw instead of the plain
    ``draw_disjoint``.

    **Access-control note (design doc §2.3):** reading ``test_ids`` here is legitimate ONLY for
    the final confirmatory scoring pass that produces a reported number -- never for arm
    selection, prompt search, threshold tuning, N*-budget choice, or reward-model calibration (see
    ``_repro/LOCKED_HOLDOUT/README.md``). This function itself does not enforce that (process
    convention + append-only ``ACCESS_LOG.md``, not a mechanical gate, per the owner's chosen
    "no encryption" tradeoff) -- the caller is responsible for only invoking ``--slice
    locked-group`` from a genuine confirmatory pass.

    Raises if any locked id is missing from a fresh pool refetch (pool drift since
    ``locked_split.py`` froze the manifest), same discipline as ``_load_rows_disjoint``.
    """
    frozen_ids = _locked_manifest_ids(dataset_key, split)
    frozen_set = set(frozen_ids)
    with _suppress_loader_snapshot_writes():
        rows = _load_rows(dataset_key, split, n=None, seed=POOL_RECONSTRUCTION_SEED,
                          restrict_ids=frozen_set)
    by_id = {r["meta"]["item_id"]: r for r in rows}
    missing = [i for i in frozen_ids if i not in by_id]
    if missing:
        raise RuntimeError(
            f"{dataset_key}/{split}: {len(missing)}/{len(frozen_ids)} locked id(s) from "
            f"LOCKED_HOLDOUT/{dataset_key}.json not found in a fresh full-pool refetch (first few: "
            f"{missing[:5]}) -- pool drift since locked_split.py froze this manifest?"
        )
    return [by_id[i] for i in frozen_ids]


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

def _run_item(dataset_key: str, backbone: str, row: dict, item_seed: int) -> dict:
    """Build the instruction, generate, and score exactly ONE row -- the per-item unit of work
    shared by both of ``run_one``'s scheduling paths below (sequential ``for`` loop when
    ``parallel<=1``, ``ThreadPoolExecutor`` when ``parallel>1``, 2026-07-10 wave-3 batched-inference
    task brief item 1). Factored out unchanged from the old inline loop body so scoring is byte-for-
    byte identical either way -- only the SCHEDULING differs, never the per-item semantics (one bad
    item's exception is caught here and turned into an ``{"item_id","error"}`` entry, exactly as the
    old inline ``try/except`` did, never sinking the whole cell).

    2026-07-11 (ticket #26, group-split design doc §2.5, "group labels aren't stored today"):
    every returned entry ALSO carries ``"group_id"`` (``group_key_of(dataset_key, row)`` -- a str,
    or None for a G-SOURCE/G-NONE dataset, see ``scripts/loaders/group_key.py``). Computed OUTSIDE
    the generation/scoring try/except (a pure function over already-loaded ``row`` data, no model
    call) but itself guarded so a bug in ``group_key_of`` can never sink a cell's scoring -- worst
    case ``group_id`` silently degrades to None, exactly the honest "no group" signal a G-NONE
    dataset would produce anyway. This is purely ADDITIVE (new dict key); nothing today reads it
    back, so every existing consumer of a per_item entry is unaffected."""
    try:
        group_id = group_key_of(dataset_key, row)
    except Exception:  # noqa: BLE001 -- group_id is provenance, never allowed to sink a cell
        group_id = None
    try:
        instr = templates.build_instruction(dataset_key, row)
        text = generate(backbone, row["wav"], instr, seed=item_seed)
        result = metrics.score(dataset_key, row, text)
    except Exception as e:  # noqa: BLE001 -- one bad item must not sink the whole cell
        return {"item_id": row.get("meta", {}).get("item_id"),
                "error": f"{type(e).__name__}: {str(e)[:200]}", "group_id": group_id}
    return {"item_id": row["meta"]["item_id"], "instr": instr, "reply": text,
            "score": result["score"], "detail": result["detail"], "group_id": group_id}


def run_one(dataset_key: str, backbone: str, split: str, n: int | None, parallel: int = 1,
            slice_mode: str = "seeded") -> dict:
    """Run one (dataset, backbone, split) cell.

    ``slice_mode`` (2026-07-10 dev/test disjoint-redraw task, owner ruling Decision-Log 续10):
    ``"seeded"`` (default) is the ORIGINAL, unchanged behavior -- draw ``n`` rows via
    ``DEV_SEED``/``TEST_SEED`` (independently of the other split, which is exactly what caused
    dev/test overlap). ``"disjoint"`` instead loads the frozen, genuinely-disjoint id slice
    ``scripts/baselines/redraw.py`` produced (see ``_load_rows_disjoint``) -- ``n`` is then IGNORED
    (any caller-supplied override is meaningless once the slice is frozen) and the cell's actual
    row count comes from the manifest itself (may be < DEV_N/TEST_N for a small-pool dataset that
    hit ``draw_disjoint``'s proportional-shortfall path). ``"locked-group"`` (2026-07-11, ticket #26
    task 5 -- PREP ONLY, see ``scripts/baselines/run_locked_rerun.sh``) loads the group-aware
    ``LOCKED_HOLDOUT`` manifest instead (see ``_load_rows_locked``) -- same "``n`` is ignored,
    frozen manifest count wins" contract as ``"disjoint"``.

    ``parallel`` (2026-07-10, wave-3 batched-inference task brief item 1): default 1 reproduces the
    ORIGINAL sequential ``for`` loop byte-for-byte (every wave-1/wave-2 caller, unaffected). ``>1``
    dispatches each item as an INDEPENDENT HTTP request to the resident llama-server via
    ``concurrent.futures.ThreadPoolExecutor`` -- valid because each request is its own isolated
    generation call (no shared mutable state across items); results are reassembled in the
    ORIGINAL row-index order regardless of HTTP completion order, so ``per_item``/aggregate output
    is identical in content and order to a sequential run, only faster wall-clock. Per-item
    try/except semantics are unchanged either way (see ``_run_item``). Owner-approved smoke verdict
    backing this (2026-07-10): server ``-np 4 -c 16384``, mtmd-compatible, 0/24 greedy flips vs
    sequential, VRAM ~20.9GB, 1.75x wall-clock speedup at K=4 (no prompt-cache reuse). ``parallel``
    should not exceed the resident server's own ``-np`` slot count (see
    ``server_slot_config_for``) -- oversubscribing just queues extra requests server-side.
    """
    seed = DEV_SEED if split == "dev" else TEST_SEED  # still used for the generation seed offset
    # (item_seed = seed + i, below) + result-JSON provenance even in slice_mode="disjoint" -- only
    # WHICH ROWS get loaded changes between the two slice modes, not the per-item generation seed.

    assert_gpu_session_held(BACKBONES[backbone].get("gpu_session_name", backbone))

    if slice_mode == "disjoint":
        rows = _load_rows_disjoint(dataset_key, split)
        n = len(rows)  # frozen manifest's actual count (may be < DEV_N/TEST_N, see draw_disjoint
        # shortfall handling) overrides any caller-supplied --n, which would be meaningless once
        # the slice is already frozen.
    elif slice_mode == "locked-group":
        rows = _load_rows_locked(dataset_key, split)
        n = len(rows)  # same "frozen manifest count wins" contract as slice_mode="disjoint"
    elif slice_mode == "seeded":
        n = n if n is not None else (DEV_N if split == "dev" else TEST_N)
        rows = _load_rows(dataset_key, split, n, seed)
    else:
        raise ValueError(
            f"run_one: unknown slice_mode {slice_mode!r} (expected 'seeded', 'disjoint', or 'locked-group')")

    kt = templates.k_type_of(dataset_key) if dataset_key not in templates.LEGACY_DATASETS else "K8/K6 (legacy)"

    t0 = time.time()
    if parallel <= 1:
        per_item = []
        for i, row in enumerate(rows):
            per_item.append(_run_item(dataset_key, backbone, row, seed + i))
            if (i + 1) % 20 == 0:
                print(f"  [{dataset_key}/{backbone}/{split} {i+1}/{len(rows)}] "
                      f"({time.time()-t0:.0f}s)", flush=True)
    else:
        import concurrent.futures

        per_item = [None] * len(rows)
        n_done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(_run_item, dataset_key, backbone, row, seed + i): i
                    for i, row in enumerate(rows)}
            for fut in concurrent.futures.as_completed(futs):
                per_item[futs[fut]] = fut.result()
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  [{dataset_key}/{backbone}/{split} {n_done}/{len(rows)} parallel={parallel}] "
                          f"({time.time()-t0:.0f}s)", flush=True)

    scores = [it["score"] for it in per_item if "score" in it]  # error-only entries have no "score" key
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

    if slice_mode == "disjoint":
        snapshot_ref = f"SPEECHRL_KB_DIR/snapshots/{_disjoint_manifest_name(dataset_key, split)}/sample_manifest.json"
    elif slice_mode == "locked-group":
        snapshot_ref = f"_repro/LOCKED_HOLDOUT/{dataset_key}.json"
    else:
        snapshot_ref = f"SPEECHRL_KB_DIR/snapshots/baselines-{dataset_key}/sample_manifest.json" \
            if dataset_key in templates.LEGACY_DATASETS else \
            f"SPEECHRL_KB_DIR/snapshots/{dataset_key}/sample_manifest.json"

    out = {
        "stage": "1-directional",
        "dataset": dataset_key, "k_type": kt, "backbone": backbone, "split": split,
        "slice_mode": slice_mode,
        "n_requested": n, "seed": seed,
        "sample_manifest": snapshot_ref,
        "model": BACKBONES[backbone]["model_tag"],
        "sampling_params": {
            "greedy_seed_base": seed, **sampling_params_for(backbone),
            # 2026-07-10 wave-3 batched-inference task brief item 1: n_parallel is the CLIENT-side
            # concurrent-request count this run_one() call used (1 = unchanged sequential);
            # server_slots/cache_ram_mib are the resident llama-server's OWN -np/--cache-ram launch
            # flags, read back from the exact env vars gpu_session.sh used to start it (see
            # server_slot_config_for) -- not independently re-measured, so they reflect the serve
            # config actually in effect, not a guess.
            "n_parallel": parallel, **server_slot_config_for(backbone),
        },
        "boundary": ("audio-only input + fixed task-definition text (label set / MCQ options / "
                     "tool registry / JSON schema); no golden transcript/answer/intent ever placed "
                     "in the prompt -- see templates.py module docstring's Information-Boundary-Guard note."),
        # ticket #25 P4: shared provenance block (git SHA+dirty, engine build id, dataset
        # revision, manifest hash, env versions) -- see provenance.collect_provenance's docstring.
        # No KB source is involved in a Stage-1 baseline cell, so manifest_hash/dataset_revision
        # stay None here (unlike run_mock's result writer, which resolves them from the KB
        # source's own manifest).
        "provenance": provenance.collect_provenance(REPO_ROOT),
        "per_item": per_item,
        "aggregate": aggregate,
        "reproduce": (
            "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data "
            "SPEECHRL_KB_DIR=/mnt/e/speechrl-knowledge python scripts/baselines/run_baseline.py "
            f"--dataset {dataset_key} --backbone {backbone} --split {split}"
            + (f" --n {n}" if slice_mode == "seeded" else "")
            + (f" --parallel {parallel}" if parallel != 1 else "")
            + (" --slice disjoint" if slice_mode == "disjoint" else "")
            + (" --slice locked-group" if slice_mode == "locked-group" else "")
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
        if dataset_key not in templates.K4_LABEL_SETS:
            # vocalbench-emotion is the only K4 grid key that is NOT in templates.K4_LABEL_SETS --
            # its label set instead comes from meta["_label_set"], populated for real by
            # run_baseline._load_rows's dedicated branch (2026-07-10 fix, see label_inventories.py's
            # VOCALBENCH_EMOTION_EMOTIONS). _synthetic_row never calls _load_rows (see this
            # function's docstring), so a --dry-run preview needs its own synthetic population here
            # too, or the preview would (again) hit templates.build_instruction's now-loud "missing
            # label set" raise instead of showing the real 5 options -- exactly the regression this
            # fix is meant to catch.
            meta["_label_set"] = label_inventories.VOCALBENCH_EMOTION_EMOTIONS
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


def _disjoint_n_preview(dk: str, split: str) -> str:
    """CPU-metadata-only preview of a frozen disjoint manifest's actual row count -- reads
    ``load_snapshot_ids`` (a plain JSON read) ONLY, never a loader/pool reconstruction, so
    --dry-run --slice disjoint stays a true zero-model/zero-heavy-IO preview (2026-07-10 dev/test
    disjoint-redraw task, verify step: "--dry-run of run_baseline --slice disjoint ... rendering
    the right item count")."""
    try:
        n = len(load_snapshot_ids(_disjoint_manifest_name(dk, split)))
        return str(n)
    except Exception as e:  # noqa: BLE001 -- dry-run must never crash on a not-yet-redrawn dataset
        return f"<no disjoint manifest: {type(e).__name__}>"


def _locked_n_preview(dk: str, split: str) -> str:
    """CPU-metadata-only preview of a LOCKED_HOLDOUT manifest's actual row count -- reads
    ``locked_split.load_locked_test``/``load_locked_dev`` (a plain JSON read) ONLY, mirroring
    ``_disjoint_n_preview``'s zero-model/zero-heavy-IO discipline (2026-07-11, ticket #26 task 5
    prep)."""
    try:
        import locked_split

        ids = locked_split.load_locked_test(dk) if split == "test" else locked_split.load_locked_dev(dk)
        return str(len(ids))
    except Exception as e:  # noqa: BLE001 -- dry-run must never crash on a not-yet-locked dataset
        return f"<no locked manifest: {type(e).__name__}>"


def dry_run(datasets: list[str], backbones: list[str], splits: list[str],
            slice_mode: str = "seeded") -> None:
    print(f"=== Stage-1 Step-1 baseline grid: {len(datasets)} datasets x {len(backbones)} backbones "
          f"x {len(splits)} splits = {len(datasets) * len(backbones) * len(splits)} cells "
          f"[slice_mode={slice_mode}] ===\n")
    for dk in datasets:
        kt = "legacy" if dk in templates.LEGACY_DATASETS else templates.k_type_of(dk)
        try:
            instr = templates.build_instruction(dk, _synthetic_row(dk))
            preview = instr.replace("\n", " \\n ")[:100]
        except Exception as e:  # noqa: BLE001 -- dry-run must never crash on one bad template
            preview = f"<template error: {type(e).__name__}: {e}>"
        if slice_mode == "disjoint":
            n_dev, n_test = _disjoint_n_preview(dk, "dev"), _disjoint_n_preview(dk, "test")
        elif slice_mode == "locked-group":
            n_dev, n_test = _locked_n_preview(dk, "dev"), _locked_n_preview(dk, "test")
        else:
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
    # 2026-07-10 wave-3 batched-inference task brief item 1: default 1 = unchanged sequential loop
    # (every wave-1/wave-2 invocation, byte-for-byte identical to before this flag existed). >1 uses
    # concurrent.futures.ThreadPoolExecutor over items -- see run_one's docstring for the owner-
    # approved smoke verdict this is based on. Should not exceed the resident server's own -np slot
    # count (gpu_session.sh's LLAMA_NP/MERALION_NP; see server_slot_config_for).
    ap.add_argument("--parallel", type=int, default=1,
                     help="client-side concurrent HTTP requests per cell (default 1 = sequential)")
    # 2026-07-10 dev/test disjoint-redraw task (owner ruling Decision-Log 续10): "seeded" (default)
    # is the ORIGINAL DEV_SEED/TEST_SEED-independent-draw behavior, byte-for-byte unchanged for
    # every wave-1/wave-2/Phase-A caller that never passes --slice. "disjoint" loads the frozen,
    # genuinely-disjoint slice scripts/baselines/redraw.py produced -- see run_one's docstring.
    ap.add_argument("--slice", dest="slice_mode", default="seeded",
                     choices=["seeded", "disjoint", "locked-group"],
                     help="'seeded' (default, unchanged), 'disjoint' (frozen redraw.py manifest -- "
                          "permanently non-locked, see _repro/redraw_manifest.NONLOCKED.md), or "
                          "'locked-group' (frozen scripts/baselines/locked_split.py LOCKED_HOLDOUT "
                          "manifest, ticket #26 task 5 -- PREP ONLY, see run_locked_rerun.sh)")
    args = ap.parse_args()

    datasets = _all_dataset_keys() if args.dataset == "ALL" else [args.dataset]
    backbones = list(BACKBONES.keys()) if args.backbone == "ALL" else [args.backbone]

    if args.dry_run:
        dry_run(datasets, backbones, [args.split], slice_mode=args.slice_mode)
        return

    for dk in datasets:
        for bb in backbones:
            print(f"=== running {dk} / {bb} / {args.split} [slice={args.slice_mode}] ===", flush=True)
            try:
                result = run_one(dk, bb, args.split, args.n, parallel=args.parallel,
                                  slice_mode=args.slice_mode)
            except Exception as e:
                print(f"  !! {dk}/{bb}/{args.split} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            p = write_result(result)
            print(f"  wrote {p} (mean={result['aggregate']['mean']} ci={result['aggregate']['ci95']})",
                  flush=True)


if __name__ == "__main__":
    main()
