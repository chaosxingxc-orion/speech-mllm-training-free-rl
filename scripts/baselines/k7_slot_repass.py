"""scripts/baselines/k7_slot_repass.py — 2026-07-10 freeze-repair CPU-only K7 aggregation repass.

Context (mirrors rescore_cells.py's own "2026-07-10 freeze-repair" framing -- read that script
first, this one follows its precedent exactly): K7 (SLU slot-filling)'s per-item ``metrics.score``
returns ``score=None`` BY DESIGN -- slot-F1 is a CORPUS-level metric that needs the whole cell's
predictions at once (``metrics.aggregate_slot_f1_slurp`` / ``aggregate_slot_f1_slue``). Both
aggregators were already WIRED (not reimplemented) before this pass, but FREEZE_SHEET.md's own K7
row flagged them as "wired but **not executed** in this draft" -- so every K7 result JSON's
``aggregate`` was just ``{"mean": null, "ci95": null, "n_scored": 0, "n_total": N, "note": "..."}``,
never an actual F1 number, across all 6 stored K7 cells (slurp-slot, speech-massive-de-DE-slot,
speech-massive-fr-FR-slot x dev/test, qwen3-omni-30b-gguf only -- meralion-2-gguf's K7 cells have
not been generated yet, wave-2 GPU execution gate).

This script closes that gap WITHOUT regenerating anything -- every K7 result JSON already stores
each item's model reply's ``detail["parsed_slots"]`` (see ``metrics.parse_k7_slots``). It re-derives
each cell's gold rows via the SAME ``(split, n_requested, seed)`` recorded in that result JSON by
re-invoking ``run_baseline._load_rows`` (reused, not reimplemented -- the SAME row-loading
dispatcher ``run_baseline.run_one`` itself calls, so a repass's rows are provably the exact rows
the original generation pass used, mirrors rescore_cells.py's own reuse of this exact function),
builds gold/pred span lists from the STORED ``parsed_slots`` + re-derived gold, and calls the
ALREADY-WIRED ``metrics.aggregate_slot_f1_slurp`` (slurp-slot, subprocess to the vendored SLURP
CLI) / ``metrics.aggregate_slot_f1_slue`` (speech-massive-*-slot, imports slue-toolkit's
``get_ner_scores``) to compute each cell's aggregate.

Two scoring conventions, reported side-by-side per the task brief ("your SLURP/slue scorers may
differ by partial-credit convention; report both"):
  - ``exact_match_micro_f1`` -- EXACT (type, phrase) match, comparable across all 3 datasets (for
    slurp-slot this is the SLURP CLI's own "entities" table, i.e. ``SpanFMeasure``'s exact dict
    match -- NOT the CLI's headline "SLU F1", which blends word/char EDIT-DISTANCE partial credit
    into precision/recall, a materially different, more forgiving convention; for
    speech-massive-*-slot this IS the slue-toolkit ``get_ner_scores`` micro F1, since that scorer
    is exact-match-only by construction). This is the number cross-checked against the audit's
    independent reference (fr-FR-slot/test ~= 0.207, de-DE-slot/dev ~= 0.247).
  - ``slurp_slu_f1_partial_credit`` (slurp-slot only) -- the SLURP CLI's own headline "SLU F1"
    (mediates word-distance-F1 and char-distance-F1, per the SLURP paper), kept for the record
    since it IS the metric the SLURP authors consider canonical, just not the "exact-match"
    convention this repass's cross-check is stated against.

Phrase normalization: both gold and predicted slot VALUES are lowercased+stripped (``_norm_phrase``)
before comparison -- for slurp-slot this MATCHES the vendored ``util.release2prediction``'s own
gold-filler convention (``token["surface"].lower()``) exactly, so applying it symmetrically to our
own predicted fillers (which are NOT otherwise lowercased, being raw model JSON output) avoids
under-counting matches on pure case difference. For speech-massive-*-slot we build BOTH sides
ourselves, so the same convention is applied uniformly rather than adding a second normalization
rule -- this is intentionally lighter than ``metrics.norm`` (no NFKC/punctuation-stripping) to stay
as close as possible to the SLURP CLI's own comparison semantics.

Every REWRITTEN file gets a ``"rescored"`` provenance block (mirrors rescore_cells.py's exact
shape: ``{"date", "reason", "original_aggregate"}``) so the previously-null aggregate is never
silently lost. ``--check-only`` recomputes and prints without writing.

Usage:
  python scripts/baselines/k7_slot_repass.py --check-only     # validate all 6 cells, no write
  python scripts/baselines/k7_slot_repass.py                  # write all 6 cells for real
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/baselines

import metrics                          # noqa: E402  (just-fixed K4/K7 scorers)
import run_baseline as rb               # noqa: E402  (reuse _load_rows -- adds scripts/loaders to sys.path)
from _common import data_root           # noqa: E402  (scripts/loaders/_common.py)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = REPO_ROOT / "_repro" / "baselines"

REPAIR_DATE = "2026-07-10"
REPAIR_REASON = (
    "K7 CPU aggregation repass (wave-1 audit): metrics.aggregate_slot_f1_slurp / "
    "aggregate_slot_f1_slue were wired (see metrics.py) but never actually invoked by "
    "run_baseline.run_one -- every K7 cell's aggregate was {mean:null, n_scored:0, ...} with no "
    "corpus-level F1 (FREEZE_SHEET.md's own 'wired but NOT executed' caveat). This repass "
    "recomputes corpus-level slot-F1 from the ALREADY-STORED per-item parsed_slots + rows "
    "re-derived via run_baseline._load_rows(dataset, split, n_requested, seed) -- NO "
    "regeneration. See k7_slot_repass.py's module docstring for the exact_match_micro_f1 vs "
    "slurp_slu_f1_partial_credit distinction and the phrase-normalization convention used."
)

# The 6 stored K7 cells as of this repair (qwen3-omni-30b-gguf only -- meralion-2-gguf's K7 cells
# are not generated yet, wave-2 GPU execution gate; see run_wave2.sh/wave2_cells.py).
K7_CELLS: list[tuple[str, str, str]] = [
    ("slurp-slot", "qwen3-omni-30b-gguf", "dev"),
    ("slurp-slot", "qwen3-omni-30b-gguf", "test"),
    ("speech-massive-de-DE-slot", "qwen3-omni-30b-gguf", "dev"),
    ("speech-massive-de-DE-slot", "qwen3-omni-30b-gguf", "test"),
    ("speech-massive-fr-FR-slot", "qwen3-omni-30b-gguf", "dev"),
    ("speech-massive-fr-FR-slot", "qwen3-omni-30b-gguf", "test"),
]


def cell_path(dataset: str, backbone: str, split: str, baselines_dir: Path = BASELINES_DIR) -> Path:
    return baselines_dir / f"{dataset}__{backbone}__{split}.json"


def _norm_phrase(s) -> str:
    """Lightest-touch normalization for slot VALUE text -- see module docstring's rationale
    (mirrors the vendored SLURP util.release2prediction's own ``.lower()`` gold-filler
    convention; deliberately NOT the heavier ``metrics.norm`` NFKC/punctuation-stripping used
    elsewhere in this repo, to stay close to the wired scorers' own comparison semantics)."""
    return str(s).strip().lower()


def _dedup_tuples(pairs: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """``[(label, phrase), ...]`` -> ``[(label, phrase, dup_id), ...]`` -- ``get_ner_scores``'s
    input contract needs a 3rd "identifier" element to differentiate REPEATED (label, phrase)
    pairs within the same item (see eval_utils_ner.get_ner_scores's own docstring example);
    dup_id is just the running occurrence count of that exact (label, phrase) pair so far."""
    seen: dict[tuple[str, str], int] = {}
    out = []
    for label, phrase in pairs:
        key = (label, phrase)
        idx = seen.get(key, 0)
        seen[key] = idx + 1
        out.append((label, phrase, idx))
    return out


def _bio_free_spans(tokens: list[str], labels: list[str]) -> list[tuple[str, str]]:
    """Contiguous same-(non-"Other")-label token runs -> ``[(label, phrase), ...]``.

    speech-massive's ``labels`` are FLAT, not BIO-prefixed (e.g. "Other"/"device_type" repeated
    across a span, verified against a live fr-FR row 2026-07-10 -- NOT "B-device_type"/
    "I-device_type") -- confirmed too that two same-label tokens separated by an "Other" token
    are DISTINCT entities (e.g. "france ... et ... chine" both tagged place_name are two separate
    1-token spans, not one span spanning the gap), so a plain contiguous-run walk (not a B-/I-
    state machine) is the correct extraction. Mirrors run_baseline._load_rows's own
    ``lab != "Other"`` filter for the closed slot-type list already shown to the model.
    """
    spans: list[tuple[str, str]] = []
    cur_label: str | None = None
    cur_toks: list[str] = []
    for tok, lab in zip(tokens, labels):
        if lab == cur_label and lab != "Other":
            cur_toks.append(tok)
            continue
        if cur_label is not None and cur_label != "Other":
            spans.append((cur_label, " ".join(cur_toks)))
        cur_label, cur_toks = lab, [tok]
    if cur_label is not None and cur_label != "Other":
        spans.append((cur_label, " ".join(cur_toks)))
    return spans


def _recompute_slurp_slot(data: dict) -> dict:
    dataset_key, split = data["dataset"], data["split"]
    n, seed = data["n_requested"], data["seed"]
    rows = rb._load_rows(dataset_key, split, n, seed)          # reuse, not reimplement
    row_by_id = {r["meta"]["item_id"]: r for r in rows}

    pred_by_file: dict[str, dict] = {}
    unresolved = []
    for it in data["per_item"]:
        row = row_by_id.get(it.get("item_id"))
        if row is None:
            unresolved.append(it.get("item_id"))
            continue
        rec_file = row["meta"]["recording_file"]
        slots = (it.get("detail") or {}).get("parsed_slots") or []
        pred_by_file[rec_file] = {
            "scenario": "", "action": "",  # K7's template is slot-only; no intent is elicited/scored here
            "entities": [{"type": s["type"], "filler": _norm_phrase(s["value"])}
                         for s in slots if s.get("type") and s.get("value")],
        }

    # slurp-slot ALWAYS draws from slurp's own "test" split regardless of the wave-2 dev/test cell
    # (see run_baseline._load_rows's "-slot" branch: split="test" for base_key=="slurp" always) --
    # so the SLURP CLI's gold jsonl selector is always "test" here too, for BOTH dev and test cells.
    result = metrics.aggregate_slot_f1_slurp(pred_by_file, data_root(), split="test")
    table_f1s = result["detail"].get("table_f1s") or {}
    aggregate = {
        "mean": None,  # unchanged: no per-item score exists for K7 (see run_one's own K7 note)
        "slot_f1": {
            "exact_match_micro_f1": result["detail"].get("entities_exact_f1"),
            "slurp_slu_f1_partial_credit": result["score"],
            "table_f1s": table_f1s,
            "checker": "slurp-native (repos/slurp/scripts/evaluation/evaluate.py)",
        },
        "n_scored": len(pred_by_file), "n_total": len(data["per_item"]),
    }
    if unresolved:
        aggregate["n_unresolved"] = len(unresolved)
    return {"aggregate": aggregate, "unresolved": unresolved, "raw_detail": result["detail"]}


def _recompute_speech_massive_slot(data: dict) -> dict:
    dataset_key, split = data["dataset"], data["split"]
    n, seed = data["n_requested"], data["seed"]
    rows = rb._load_rows(dataset_key, split, n, seed)          # reuse, not reimplement
    row_by_id = {r["meta"]["item_id"]: r for r in rows}

    gold_spans, pred_spans = [], []
    unresolved = []
    for it in data["per_item"]:
        row = row_by_id.get(it.get("item_id"))
        if row is None:
            unresolved.append(it.get("item_id"))
            gold_spans.append([])
            pred_spans.append([])
            continue
        gold = row["gold"]
        gold_pairs = [(lab, _norm_phrase(ph)) for lab, ph in _bio_free_spans(gold["tokens"], gold["labels"])]
        slots = (it.get("detail") or {}).get("parsed_slots") or []
        pred_pairs = [(s["type"], _norm_phrase(s["value"])) for s in slots if s.get("type") and s.get("value")]
        gold_spans.append(_dedup_tuples(gold_pairs))
        pred_spans.append(_dedup_tuples(pred_pairs))

    result = metrics.aggregate_slot_f1_slue(gold_spans, pred_spans, data_root())
    overall_micro = result["detail"].get("overall_micro") or {}
    overall_macro = result["detail"].get("overall_macro") or {}
    aggregate = {
        "mean": None,  # unchanged: no per-item score exists for K7 (see run_one's own K7 note)
        "slot_f1": {
            "exact_match_micro_f1": overall_micro.get("fscore"),
            "exact_match_macro_f1": overall_macro.get("fscore"),
            "checker": "slue-toolkit get_ner_scores",
        },
        "n_scored": len(data["per_item"]) - len(unresolved), "n_total": len(data["per_item"]),
    }
    if unresolved:
        aggregate["n_unresolved"] = len(unresolved)
    return {"aggregate": aggregate, "unresolved": unresolved, "raw_detail": result["detail"]}


def recompute_cell(path: Path) -> dict:
    """Recompute one K7 cell's aggregate from its STORED per-item parsed_slots (no regeneration,
    no write). Returns a report dict; see ``repass_cell`` for the write step."""
    data = json.loads(path.read_text(encoding="utf-8"))
    dataset_key = data["dataset"]
    if dataset_key == "slurp-slot":
        out = _recompute_slurp_slot(data)
    elif dataset_key.startswith("speech-massive-") and dataset_key.endswith("-slot"):
        out = _recompute_speech_massive_slot(data)
    else:
        raise ValueError(f"k7_slot_repass: no handler for dataset_key={dataset_key!r}")

    # idempotency: chain through the TRUE original aggregate if this file was already repassed by
    # a prior run of this script -- mirrors rescore_cells.recompute_cell's exact same discipline.
    true_original_aggregate = data.get("rescored", {}).get("original_aggregate", data.get("aggregate"))
    return {
        "path": path, "data": data, "new_aggregate": out["aggregate"],
        "old_aggregate": true_original_aggregate, "unresolved": out["unresolved"],
        "raw_detail": out["raw_detail"],
    }


def repass_cell(path: Path) -> dict:
    """``recompute_cell`` + write the result back to ``path`` with a ``"rescored"`` provenance
    block (same shape as rescore_cells.py's, so both repairs share one audit-trail convention)."""
    report = recompute_cell(path)
    data = report["data"]
    data["aggregate"] = report["new_aggregate"]
    data["rescored"] = {
        "date": REPAIR_DATE, "reason": REPAIR_REASON,
        "original_aggregate": report["old_aggregate"],
    }
    if report["unresolved"]:
        data["rescored"]["unresolved_item_ids"] = report["unresolved"]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=BASELINES_DIR)
    ap.add_argument("--check-only", action="store_true",
                     help="recompute + print but do NOT write (validation mode)")
    args = ap.parse_args()

    n_ok = n_missing = n_err = 0
    for dk, bb, sp in K7_CELLS:
        p = cell_path(dk, bb, sp, args.dir)
        if not p.exists():
            print(f"SKIP (missing file): {p.name}")
            n_missing += 1
            continue
        try:
            report = recompute_cell(p) if args.check_only else repass_cell(p)
        except Exception as e:  # noqa: BLE001 -- one bad cell must not sink the whole pass
            print(f"ERROR {p.name}: {type(e).__name__}: {e}")
            n_err += 1
            continue
        old_agg = report["old_aggregate"] or {}
        new_agg = report["new_aggregate"]
        tag = "CHECK" if args.check_only else "WROTE"
        print(f"[{tag}] {p.name}: old.mean={old_agg.get('mean')} -> "
              f"slot_f1={json.dumps(new_agg.get('slot_f1'), ensure_ascii=False)} "
              f"(n_scored={new_agg['n_scored']}/{new_agg['n_total']})")
        if report["unresolved"]:
            print(f"    WARNING: {len(report['unresolved'])} item_id(s) not found in re-derived "
                  f"rows: {report['unresolved'][:5]}")
        n_ok += 1

    print(f"\n=== {n_ok} cell(s) processed, {n_missing} missing, {n_err} error(s) ===")


if __name__ == "__main__":
    main()
