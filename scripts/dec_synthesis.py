"""DEC (#35) synthesis engine — apply the P1 frozen rule mechanically across all lever artifacts.

Reads the committed _repro artifacts (p2 baselines, E7 few-shot, E8 prompt-opt, E10 verifier) and, per
surface, computes: greedy, oracle-delta ceiling, each lever's realized deployable gain and whether it
clears the P1 bar (relative +10%, b2-genuine, transfer for E8), then the mechanical 2.1/2.2 verdict.
Pure analysis (no GPU). Run after E7/E8/E10 land; reports what is available so far.

reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/dec_synthesis.py
"""
import json, os
from pathlib import Path
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
R = DATA / "_repro"
REL_BAR = 0.10          # P1: relative +10%
RHO_MIN = 0.30          # P1: min oracle-delta fraction a selector must realize to "work"
TRANSFER_MIN = 0.70     # P1: E8 held-out/dev retained ratio


def load(name):
    p = R / name
    return json.load(open(p, encoding="utf-8")).get("results", {}) if p.exists() else {}


def main():
    p2 = load("p2_baselines.json")
    e7 = load("e7_fewshot.json")
    e8 = load("e8_promptopt.json")
    e10 = load("e10_verifier.json")
    surfaces = sorted(set(p2) | set(e7) | set(e8) | set(e10))
    rows, nonsat = [], []
    for s in surfaces:
        b = p2.get(s, {})
        greedy = b.get("greedy"); oracle = b.get("oracle"); od = b.get("oracle_delta")
        sat = b.get("saturated", None)
        row = {"surface": s, "greedy": greedy, "oracle_delta": od, "saturated": sat}
        # E7: best few-shot b2 gain over 0-shot (relative), and b2-genuine (over b1 floor)
        if s in e7 and "shots" in e7[s]:
            sh = e7[s]["shots"]; z = sh.get("0", sh.get(0, {}))
            base = z.get("greedy_b2")
            best = max(((v.get("greedy_b2"), v.get("b2_minus_b1")) for k, v in sh.items()),
                       key=lambda t: (t[0] or 0), default=(None, None))
            if base and best[0] is not None:
                row["e7_rel_gain"] = round((best[0] - base) / max(1e-9, base), 4)
                row["e7_b2_genuine"] = (best[1] or 0) > 0
        # E8: rel test gain + transfer
        if s in e8 and "rel_gain" in e8[s]:
            row["e8_rel_gain"] = e8[s]["rel_gain"]; row["e8_transfer"] = e8[s].get("transfer")
        # E10: two-system verifier rho + isolation gain
        if s in e10 and "rho_isolated" in e10[s]:
            row["e10_rho_iso"] = e10[s]["rho_isolated"]; row["e10_iso_gain"] = e10[s].get("isolation_gain_over_coupled")
        # per-surface: did any lever clear the bar?
        cleared = []
        if row.get("e7_rel_gain", -9) >= REL_BAR and row.get("e7_b2_genuine"):
            cleared.append("E7")
        if row.get("e8_rel_gain", -9) >= REL_BAR and (row.get("e8_transfer") or 0) >= TRANSFER_MIN:
            cleared.append("E8")
        if row.get("e10_rho_iso", -9) >= RHO_MIN:
            cleared.append("E10")
        row["cleared"] = cleared
        rows.append(row)
        if sat is False:
            nonsat.append(row)
    # overall (P1): a lever "works" if it clears on a MAJORITY of non-saturated surfaces
    from collections import Counter
    c = Counter()
    for r in nonsat:
        for l in r["cleared"]:
            c[l] += 1
    need = (len(nonsat) + 1) // 2
    winners = {l: n for l, n in c.items() if n >= need}
    verdict = ("2.1 (space IS training-free-optimizable via A/selection): " + ", ".join(sorted(winners))
               if winners else
               "2.2 (no in-fence lever clears the bar on a majority of non-saturated surfaces -> agentic "
               "reward/verification expansion warranted; residual = knowledge blind-spot, PARKED #37->W4)")
    out = {"criteria": {"rel_bar": REL_BAR, "rho_min": RHO_MIN, "transfer_min": TRANSFER_MIN,
                        "coverage_rule": f">= majority of non-saturated surfaces (need {need} of {len(nonsat)})"},
           "per_surface": rows, "non_saturated": [r["surface"] for r in nonsat],
           "lever_clear_counts": dict(c), "provisional_verdict": verdict,
           "note": "PROVISIONAL — final DEC requires all of E7/E8/E10 landed + strict re-review ACCEPT."}
    (R / "dec_synthesis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\nwrote", R / "dec_synthesis.json")


if __name__ == "__main__":
    main()
