# `_repro/redraw_manifest.json` — PERMANENTLY NON-LOCKED

Sidecar note (ticket #26, design doc `wiki/2026-07-11-group-split-statistics-design.md` finding
#2) — added BESIDE the JSON, which is left byte-for-byte unedited by this ticket.

`redraw_manifest.json`'s `new_test_ids` (drawn with `TEST_SEED=20261705`, the SAME seed the
already-scored wave-1/2 grid used) are **permanently non-locked**: a holdout is only a real
holdout if its membership has never informed a decision, and `TEST_SEED` had already selected
arms/prompts/thresholds before this manifest was even drawn. Disjointness from `dev` does not fix
that — reusable-holdout contamination is about REUSE of a seed that already informed a decision,
not about dev/test overlap.

**The only locked holdout is `_repro/LOCKED_HOLDOUT/` (fresh `LOCKED_TEST_SEED=611_741_209`,
group-aware, access-controlled by convention — see that directory's `README.md`).**

This note does not change `redraw_manifest.json` itself (still the accurate historical record of
the wave-1/2 disjoint redraw) — it only stamps the "do not treat this as a locked holdout" banner
next to it, per the ticket's explicit instruction not to edit the JSON.
