"""scripts/baselines/_ifeval_check.py -- rule-checker wrapper for K11 voicebench-ifeval.

Wires (does not reimplement) the vendored google-research ``instruction_following_eval`` subtree
-- see ``$SPEECHRL_DATA_DIR/repos/instruction_following_eval/PROVENANCE.txt`` for its provenance
(subtree of github.com/google-research/google-research, Apache-2.0, fetched 2026-07-09) -- to turn
a model response + the row's ``instruction_id_list``/``kwargs`` (see
``scripts/loaders/voicebench.py:load_voicebench_ifeval``) into pass/fail rule-check results.

Lazy-import discipline (CLAUDE.md): importing this module does nothing heavy and does not touch
SPEECHRL_DATA_DIR -- the subtree's absl/langdetect/nltk/immutabledict deps and the sys.path wiring
that makes ``import instruction_following_eval`` resolve only happen inside ``check_ifeval()``\'s
first call (``_load_registry()``), so ``import _ifeval_check`` alone always succeeds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REGISTRY = None  # cached instructions_registry.INSTRUCTION_DICT, set by _load_registry()


class NeedsChecker(RuntimeError):
    """Raised when the instruction_following_eval subtree isn't present/importable.

    Callers (metrics.py's K11 ifeval branch) should catch this and degrade gracefully to the
    existing score=None stub rather than crashing the whole baseline run.
    """


def _load_registry():
    """Import the vendored subtree and return ``instructions_registry.INSTRUCTION_DICT``.

    Cached after the first successful call. Raises ``NeedsChecker`` (never a bare ImportError/
    KeyError) if the subtree is missing, incomplete, or its runtime deps aren't installed -- so
    callers can catch exactly one exception type.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    env = os.environ.get("SPEECHRL_DATA_DIR")
    if not env:
        raise NeedsChecker(
            "SPEECHRL_DATA_DIR is not set -- cannot locate the instruction_following_eval subtree "
            "(expected at $SPEECHRL_DATA_DIR/repos/instruction_following_eval; see CLAUDE.md "
            "Environment section)."
        )
    repos_dir = Path(env) / "repos"
    subtree = repos_dir / "instruction_following_eval"
    if not (subtree / "instructions_registry.py").is_file():
        raise NeedsChecker(
            f"instruction_following_eval subtree not found at {subtree} (expected "
            "instructions_registry.py there) -- see that dir's PROVENANCE.txt for the expected "
            "fetch, or this module's docstring."
        )

    if str(repos_dir) not in sys.path:
        sys.path.insert(0, str(repos_dir))
    try:
        # No __init__.py in the vendored subtree -- resolves as a PEP 420 implicit namespace
        # package as long as `repos_dir` (the subtree's PARENT) is on sys.path, matching how
        # evaluation_lib.py itself does `from instruction_following_eval import ...`.
        from instruction_following_eval import instructions_registry
    except ImportError as e:
        raise NeedsChecker(
            f"instruction_following_eval subtree present at {subtree} but failed to import -- "
            f"missing runtime dep? (PROVENANCE.txt's Runtime deps note: absl, langdetect, nltk, "
            f"immutabledict): {e}"
        ) from e

    _REGISTRY = instructions_registry.INSTRUCTION_DICT
    return _REGISTRY


def check_ifeval(response: str, instruction_id_list: list, kwargs: list[dict]) -> dict:
    """Rule-check ``response`` against IFEval's per-instruction checkers.

    STRICT variant only (mirrors upstream ``evaluation_lib.test_instruction_following_strict``,
    minus its prompt_to_response dict plumbing -- this call site already has exactly one response
    in hand). Loose-variant (whitespace/markdown-stripping retries) is NOT implemented; can be
    added later if strict proves too punishing for a spoken-response setting.

    Args:
        response: the model's text reply.
        instruction_id_list: e.g. ``["keywords:existence", "length_constraints:number_words"]``
            (row's ``meta["instruction_id_list"]`` from ``load_voicebench_ifeval``).
        kwargs: per-instruction kwarg dicts, same order/length as ``instruction_id_list`` (row's
            ``meta["kwargs"]``).

    Returns:
        ``{"passes": list[bool], "all_pass": bool, "frac": float}`` -- one bool per
        ``instruction_id_list`` entry (same order); ``all_pass = all(passes)``;
        ``frac = mean(passes)`` (``1.0`` for an empty instruction list -- vacuously all-followed,
        matching upstream's ``all([])`` convention).

    Raises:
        NeedsChecker: if the vendored subtree isn't present/importable. Callers MUST catch this
        and degrade gracefully (see ``metrics.py``'s K11 ifeval branch) rather than let a missing
        offline asset crash the whole baseline run.
    """
    registry = _load_registry()

    if not instruction_id_list:
        return {"passes": [], "all_pass": True, "frac": 1.0}

    passes = []
    for idx, instruction_id in enumerate(instruction_id_list):
        instruction_cls = registry[instruction_id]
        instruction = instruction_cls(instruction_id)
        kw = kwargs[idx] if idx < len(kwargs) and kwargs[idx] else {}
        # Upstream's own dataset ships only non-null kwargs per instruction; filtering None
        # explicitly here is a no-op in the common case but guards against a stray null surviving
        # parquet round-tripping (build_description's own defaults are already None, so passing
        # None explicitly vs. omitting the key is behaviorally identical -- this is belt-and-braces).
        kw = {k: v for k, v in kw.items() if v is not None}
        instruction.build_description(**kw)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            # A handful of checkers (e.g. combination:repeat_prompt) need the ORIGINAL prompt
            # text, which this function's signature does not accept (task-scoped to
            # response/instruction_id_list/kwargs only). Fail closed (not silently mis-scored) --
            # revisit by threading `prompt` through if/when this instruction type is actually
            # sampled into a dev/test draw.
            passes.append(False)
            continue
        ok = bool(response.strip()) and bool(instruction.check_following(response))
        passes.append(ok)

    frac = sum(passes) / len(passes)
    return {"passes": passes, "all_pass": all(passes), "frac": frac}


if __name__ == "__main__":
    # Smoke test (see step 3 of the closeout task) -- 3 synthetic cases, no network/audio, run
    # with: SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data
    #       python scripts/baselines/_ifeval_check.py
    cases = [
        ("length_constraints:number_words -- pass (>=3 words, have 5)",
         "one two three four five", ["length_constraints:number_words"],
         [{"num_words": 3, "relation": "at least"}], True),
        ("length_constraints:number_words -- fail (>=10 words, have 5)",
         "one two three four five", ["length_constraints:number_words"],
         [{"num_words": 10, "relation": "at least"}], False),
        ("keywords:existence -- pass (both keywords present)",
         "The quick brown fox jumps over the lazy dog.", ["keywords:existence"],
         [{"keywords": ["fox", "dog"]}], True),
    ]
    for name, response, ids, kws, expect_all_pass in cases:
        result = check_ifeval(response, ids, kws)
        status = "OK" if result["all_pass"] == expect_all_pass else "MISMATCH"
        print(f"[{status}] {name}: {result}")
