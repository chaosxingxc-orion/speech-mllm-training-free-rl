# Testing — the one standard command

Ticket #38 item 5 (2026-07-13): this repo's tests must collect and pass under a **single**
standard invocation, with zero pytest collection errors. That command is:

```bash
cd projects/speech-mllm-training-free-rl
export SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data   # WSL path
PYTHONPATH=src pytest -q
```

Run from inside the WSL2 environment (`wsl -d Ubuntu-24.04`, venv active — see the umbrella
`CLAUDE.md`'s "Environment" section), e.g. the full one-liner used by CI/agents:

```bash
wsl -d Ubuntu-24.04 bash -c '
  source ~/.venvs/speechrl/bin/activate &&
  export SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data &&
  cd /mnt/d/chao_workspace/exploring-l4-intelligence/projects/speech-mllm-training-free-rl &&
  PYTHONPATH=src pytest -q
'
```

## What "green" means

- **Zero errors, zero failures.** `pytest -q`'s summary line reads `N passed` — no `error`, no
  `failed`, no `skipped` unless a skip is explicitly justified in that test's own code.
- As of this ticket (#38, updated 2026-07-13 — V1-1/V1-2 remediation), the standard command
  collects and passes **143 tests, 0 errors** across:

  | File | Tests |
  |---|---|
  | `scripts/baselines/test_deterministic_draw.py` | 30 |
  | `scripts/baselines/test_phase_a_e2e.py` | 4 |
  | `scripts/baselines/test_stats.py` | 41 |
  | `scripts/baselines/test_two_pass_runner.py` | 27 |
  | `scripts/knowledge/test_build_full_corpus.py` | 5 |
  | `scripts/knowledge/test_kb_gate.py` | 1 |
  | `scripts/knowledge/test_knowledge_card.py` | 13 |
  | `scripts/knowledge/test_pseudo_question.py` | 21 |
  | `tests/test_smoke.py` | 1 |

  Run `PYTHONPATH=src pytest -q --collect-only` to reproduce/refresh this table — it is not
  regenerated automatically and can drift as tests are added (e.g. the group-aware confirmatory-draw
  tests item 4 of this same ticket added to `test_deterministic_draw.py`, 21 → 30); the invariant
  that must never drift is **0 errors**.

  **Follow-up closed (2026-07-13, V1-1 fix):** `scripts/knowledge/test_kb_gate.py` used to define
  **zero** top-level `def test_*` functions — its ~57 `check(label, cond, results)` assertions
  (KB leakage-gate / cross-modal-routing coverage) all lived inside a monolithic `main()`, runnable
  only via `python -u scripts/knowledge/test_kb_gate.py`, invisible to `pytest --collect-only` (0
  items collected, no error/warning either). This was the SAME underlying failure mode as the
  `test_phase_a_e2e.py` fix documented below, just further along the spectrum (whole-file, not 4
  functions). It now follows the identical pattern: the body was extracted into
  `_check_kb_gate(results: dict)` (unchanged logic) behind one zero-argument
  `test_kb_gate_all_cases()` wrapper — collected as a single pytest item (all ~57 sub-checks
  collapse to one assert whose failure message names every failing check by label), while
  `python -u scripts/knowledge/test_kb_gate.py`'s detailed per-check console report and
  `KB_GATE_TEST_PASS` sentinel are unchanged. Fixing this surfaced a real, previously-latent
  cross-file test-pollution bug: `test_phase_a_e2e.py`'s `_check_e2e_all_arms` monkeypatched
  `kb_embed.embed_audio`/`embed_text` module-globally without restoring them, which — once
  `test_kb_gate.py` became collectible and started running in the SAME pytest process right after
  it — silently fed `test_kb_gate.py` a 16-dim fake embedding space instead of the real one,
  breaking its dim-consistency check. Fixed alongside (both functions now restored in the existing
  `finally` block).

- `SPEECHRL_DATA_DIR` must be exported (even though most of these suites are offline/synthetic —
  a handful of modules read it at import time, e.g. anything importing `p2_baselines`/
  `registry`). Without it, some tests fail at collection/import time with an unrelated
  `RuntimeError`/`KeyError`, not a real test failure — always export it first.
- Every test file here is **also** directly runnable standalone
  (`python -u scripts/baselines/test_phase_a_e2e.py`, etc.) — this repo's convention (see each
  file's own docstring) is that a bare `test_*()` function is both pytest-collectible AND callable
  from that file's own `main()`, which prints a per-check PASS/FAIL summary and a final
  `..._TEST_PASS` / `..._TEST_FAIL` sentinel line. Both entries must stay green; neither is a
  substitute for the other — pytest is the CI/coordinator-facing signal, the standalone entry is
  useful for verbose, per-check debugging.

## Why "PYTHONPATH=src pytest -q" specifically

`PYTHONPATH=src` matches this repo's package layout (`src/training_free_rl`) and the sibling
work-repos' shared convention (each work's own `docs/TESTING.md`-equivalent uses the same
invocation shape against its own `src/<pkg>`) — kept for consistency even though the
`scripts/baselines`/`scripts/knowledge` test files each already insert their own directory onto
`sys.path` at import time (see each file's `HERE = os.path.dirname(...)` / `sys.path.insert`
preamble) and so do not themselves depend on `PYTHONPATH`. Do not drop `PYTHONPATH=src` from the
standard command even though today's suites don't strictly need it — a future test importing
`training_free_rl` directly should not have to rediscover this.

## History: the fixture-not-found regression this doc responds to

Before this ticket, `PYTHONPATH=src pytest -q` reported `124 passed, 4 errors` — the 4 errors were
`scripts/baselines/test_phase_a_e2e.py`'s `test_e2e_all_arms` / `test_cpu_live_logmel_stats_smoke`
/ `test_e2e_control_arms` / `test_source_name_for_squtr_corpus_routing`, each declared as
`def test_x(results: dict) -> None`. pytest treats every bare positional parameter on a
`test_*` function as a **fixture request** — since no fixture named `results` is registered, all
four errored at *setup*, before their bodies ever ran (`fixture 'results' not found`). This was a
real gap: `pytest -q`'s exit code was still 0-ambiguous-looking in casual reads ("124 passed" reads
fine at a glance) while 4 whole test files' worth of checks silently never executed under the
"standard" pytest entry — only the file's own `python -u ...` `__main__` entry actually ran them.

**Fix** (ticket #38 item 5): each of the 4 test bodies was renamed to a private
`_check_x(results: dict) -> None` (unchanged logic — still populates a rich, descriptively-keyed
`results` dict, still usable standalone), and a public, **zero-argument**
`test_x() -> None` wrapper was added that builds its own local `results` dict, calls the matching
`_check_x`, and asserts every value is truthy (`assert not [k for k, v in results.items() if not
v]`, so a pytest failure message names exactly which check(s) failed). `main()` (the
`__main__` entry) now calls the `_check_x` functions directly with one shared dict, preserving the
original detailed per-check PASS/FAIL console report exactly as before. Re-verify with:

```bash
PYTHONPATH=src pytest -q scripts/baselines/test_phase_a_e2e.py    # 4 passed
python -u scripts/baselines/test_phase_a_e2e.py                    # ... PHASE_A_E2E_TEST_PASS
```

If you ever add a new `test_*` function that needs to build up a descriptive dict of sub-checks
before asserting, follow this SAME pattern (`_check_*(results: dict)` + zero-arg `test_*()`
wrapper) rather than giving the pytest-collected function itself a positional parameter — pytest
will not error loudly at your desk the way it should; it will just silently skip your checks under
the "standard" command, which is a much worse failure mode than an upfront `ImportError`.
