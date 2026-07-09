# scripts/loaders/ — dataset loaders (thin, uniform interface)

Task B0 scaffold. This package holds ONE thing: pure dataset-loading code that turns a local
dataset (`SPEECHRL_DATA_DIR/datasets/...`) into a plain list of `Row` dicts. Nothing else lives
here — no model calls, no prompt construction, no reward/metric code. Those stay in the
experiment scripts (`scripts/p2_baselines.py`, `scripts/t*.py`, ...) that consume loader output.

## Interface contract

Every dataset gets its own module `scripts/loaders/<name>.py` exposing exactly one function:

```python
def load_<name>(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list[Row]:
    ...
```

- `split` — which split to load (dataset-dependent; default `"test"`).
- `n` — number of rows to return after slicing; `None` means "all available rows in the split".
- `seed` — the RNG seed used to draw the slice; default `_common.SLICE_SEED` (`20260705`, pinned
  2026-07-05 in `p2_baselines.py`) so independent loaders sample comparably unless a caller
  explicitly asks for a different seed.
- Returns a plain `list[Row]` — see `Row` below. No generators, no dataset-library objects.

Hard rules:

- **NO model calls.** A loader must never talk to `llama-server` / an omni endpoint / any
  inference API. It only reads local files.
- **NO experiment logic.** No instruction/prompt construction, no few-shot formatting, no
  reward-guided anything. If it shapes what the model is asked to do, it doesn't belong here.
- **NO metric code.** No WER/exact-match/accuracy scoring. That's the caller's job (e.g.
  `common/src/speechrl_common/rl` or a script-local `reward()` function).
- **MUST read `SPEECHRL_DATA_DIR`** via `_common.data_root()` / `_common.datasets_dir()` — never
  hardcode a path. `data_root()` raises loudly if the env var is unset (no silent default),
  matching `kb_schema.kb_root()`'s discipline for `SPEECHRL_KB_DIR`.
- **Slice determinism** — every loader must be replayable:
  1. Seed with `_common.SLICE_SEED` (or the caller's `seed` override) via
     `numpy.random.default_rng(seed)`.
  2. Enumerate source files/rows in `sorted()` order before sampling (never rely on filesystem
     iteration order, which is not guaranteed stable — see `p2_baselines.py`'s
     `sorted((DS / "mmau-mini/data").glob(...))` pattern).
  3. Call `_common.freeze_ids(name, ids)` with the explicit item ids actually sampled, so the
     slice is provable independent of file/row drift (wraps
     `scripts/knowledge/kb_snapshot.freeze_snapshot`; see `scripts/knowledge/README.md`'s
     "Reproducibility contract").
- **Lazy imports** — heavy deps (`pyarrow`, `soundfile`, `librosa`, `numpy`, ...) go inside
  `load_<name>` (or a helper it calls), never at module top level, so `import <name>` alone stays
  cheap and `registry.py` can be imported anywhere without the ML stack installed (CLAUDE.md
  lazy-import discipline).
- **Register it.** Add `"<name>": load_<name>` to `LOADERS` in `registry.py` so `smoke_all()`
  picks it up automatically.

## Row

```python
Row = {
    "wav": ...,   # str path to a materialized wav file (preferred), or raw undecoded audio bytes
    "gold": ...,  # str or dict — reference label/answer/transcript/intent
    "meta": {...} # dict: at minimum {"dataset", "item_id", "split"}; task-shaping fields
                  # (e.g. "task", "instr", "opts") welcome, reward/metric code is not
}
```

Prefer handing back a `wav` **path** (via `_common.wav_from_bytes(raw_bytes, key, cache_dir)`
when the source stores audio as embedded bytes, e.g. a HF parquet `{"bytes": ...}` column) so
every downstream consumer sees the same shape regardless of how a given dataset ships audio.
`_common.wav_cache_dir(name)` gives each loader its own materialization cache dir under
`SPEECHRL_DATA_DIR/_repro/loader_wavs/<name>/`, so two loaders never collide on a cache key.

## Files

| file | role |
|---|---|
| `_common.py` | `SLICE_SEED`, `Row`, `data_root()` / `datasets_dir()`, `wav_from_bytes()` / `wav_cache_dir()`, `freeze_ids()` (wraps `kb_snapshot.freeze_snapshot`) |
| `registry.py` | `LOADERS: dict[str, Callable]` + `smoke_all(n=3)` — loads `n` items from every registered loader, prints `name / count / keys`, and reports (without raising) any loader that fails |
| `<name>.py` (future) | one module per dataset, each exposing `load_<name>` per the contract above |

## Smoke test

```bash
SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data \
  python -c "import sys; sys.path.insert(0, 'scripts/loaders'); import registry; registry.smoke_all()"
```

At scaffold time (`LOADERS` empty) this prints a "nothing registered yet" line and returns `{}`
without error — the contract is enforceable and CI-checkable before the first concrete loader
lands.
