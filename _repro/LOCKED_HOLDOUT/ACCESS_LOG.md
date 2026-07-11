# ACCESS_LOG.md — append-only access log for `_repro/LOCKED_HOLDOUT/`

Process convention (owner ruling, Decision-Log 续13: "file convention + append-only access log,
no encryption" — see `README.md` in this directory for the full rule). Every read of a `test_ids`
field from any manifest in this directory should append ONE line below, in this format, and
should NEVER edit or remove an existing line:

    - YYYY-MM-DDTHH:MM:SSZ | <dataset_key(s)> | <who/what> | <why — which confirmatory pass>

Do not read `test_ids` for arm selection, prompt search, threshold tuning, N*-budget choice, or
reward-model calibration — see `README.md`.

<!-- append new entries below this line -->
- 2026-07-11T12:25:34Z | ALL 65 dataset keys | locked-dev-rerun agent (group-aware locked-DEV baseline rerun, driver: scripts/baselines/run_locked_rerun.sh --split dev, commit b4707fe) | dev_ids read for dev-half baseline scoring (65 dev cells -> *__qwen3-omni-30b-gguf__dev.locked.json, ran=50+15 checkpoint-skipped, failed=0); test_ids NOT read (only len() counts via census); test half remains single-consumer reserved for the final confirmatory pass
- 2026-07-12T00:00:00Z | ALL 65 dataset keys | 2026-07-12 RI mechanical remediation, item 5 (续15 P0-2 finding) | CORRECTION to the 2026-07-11T12:25:34Z entry above: its claim "test_ids NOT read (only len() counts via census)" was INACCURATE. `scripts/baselines/locked_rerun_cells.py`'s `print_census()` called `json.loads((LOCKED_DIR / f"{dk}.json").read_text())` then `len(manifest.get("test_ids", []))` for every test cell — that materializes the full plaintext `test_ids` list into memory to take its length, which IS a read of `test_ids`, not merely a count. Separately, the `b4707fe` overlap-verification step (ticket #26: "overlap-fraction honesty metric", 61-row aishell-1 check `overlap_count`/`overlap_fraction` fields baked into each manifest) also read `test_ids` MEMBERSHIP (not just length) to compute overlap against `old_exposed_test_ids` — a second, distinct `test_ids` read never logged as such. This entry logs BOTH reconciliation reads. Fix applied same-day: a new count-only sidecar `_counts.json` (generated once by `scripts/baselines/gen_locked_holdout_counts.py`, itself the ONE logged read of `test_ids`/`dev_ids` needed to produce it) now backs `print_census()`, which no longer opens any per-dataset manifest's id lists. Per Decision-Log 续15 (owner-signed REJECT closeout): these 65 manifests are PERMANENTLY DOWNGRADED to exposed-dev-like — see `_repro/LOCKED_HOLDOUT/README.md`'s 2026-07-12 banner — regardless of this census fix; the fix improves future access hygiene but does not un-downgrade the existing manifests (plaintext test_ids + 11.2% old-test overlap + honor-system-only access already happened).
