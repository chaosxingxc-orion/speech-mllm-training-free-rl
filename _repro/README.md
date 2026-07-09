# `_repro/` — errata

**Errata (2026-07-09, task A4):** every `"reproduce"` string inside a `_repro/*.json` file committed
**before 2026-07-09** reads `SPEECHRL_DATA_DIR=<repo>/speechrl-data`. That predates the data-root move
off the D: drive — the umbrella migration (commit `dcb97f0`, 2026-07-09) relocated
`speechrl-data/` to the **E: drive** but only updated umbrella-side paths; the W1 scripts' own
`reproduce` string constants were not touched at the time, so this repo kept emitting the stale
pre-move string into every result JSON it wrote.

**The authoritative invocation, for any JSON dated before 2026-07-09, is:**

```
SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data <rest of the command as written>
```

i.e. substitute the E-drive path for the literal `<repo>/speechrl-data` text — everything else in the
`reproduce` string (script name, flags, env vars) is unaffected and should be run as written.

**Why the JSONs themselves are not being fixed:** `_repro/*.json` files are append-only research
records (per `CLAUDE.md` / `wiki/AI-Collaboration.md` convention) — they capture what was actually run
and observed at the time, including a since-superseded env convention. Rewriting the `reproduce` field
in place would silently alter a historical record. This file is the fix instead: read it as a standing
correction layered on top of every pre-2026-07-09 JSON in this directory, not as license to hand-edit
them.

Source scripts under `scripts/` were updated on 2026-07-09 to emit the E-drive path going forward, so
any `_repro/*.json` written from that date onward already carries the correct `reproduce` string
natively and needs no correction from this note.
