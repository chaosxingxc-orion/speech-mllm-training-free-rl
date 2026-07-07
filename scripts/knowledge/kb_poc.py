"""kb_poc — end-to-end PoC of the speech-keyed knowledge base (Stage-1 verification).

Validates the full pipeline on a small heysquad slice:
  1. build a SPEECH-KEYED source: key = spoken-question audio, value = context passage (external knowledge)
  2. value-side leakage audit flags it LEAKAGE (each context contains its answer); scrub -> CLEAN
  3. persisted vector index (FAISS-flat-IP or numpy-flat) + value store round-trip
  4. audio-query retrieval: embed the same spoken-question audio -> top-1 self-match rate
  5. snapshot freeze + replay -> item-for-item reproducible

Runnable anywhere (no GPU/network): audio keys use the logmel-stats fallback; faiss optional.

    SPEECHRL_DATA_DIR=<repo>/speechrl-data SPEECHRL_KB_DIR=E:/speechrl-knowledge \
        python -u scripts/knowledge/kb_poc.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)                     # scripts/knowledge
sys.path.insert(0, os.path.dirname(HERE))    # scripts/  (for t7 / p2)

import kb_build          # noqa: E402
import kb_retrieve       # noqa: E402
import kb_snapshot       # noqa: E402
import t7_rag_gate_probe as t7   # noqa: E402  (load_heysquad + _wav)

N_KB = int(os.environ.get("POC_KB", "50"))
N_EVAL = int(os.environ.get("POC_EVAL", "20"))
SEED = 20260707


def main() -> int:
    print(f"[kb_poc] kb={N_KB} eval={N_EVAL} kb_root={os.environ.get('SPEECHRL_KB_DIR')}", flush=True)
    pool = t7.load_heysquad(N_KB)
    golds = [g for it in pool for g in it["golds"]]

    records = []
    for it in pool:
        try:
            wav = t7._wav(it)
        except Exception as e:
            print(f"  skip wav {it.get('id')}: {type(e).__name__}", flush=True)
            continue
        records.append({"key_audio_ref": wav, "value": it["ctx"], "from_item_id": it["id"]})
    print(f"  built {len(records)} records", flush=True)

    # --- build raw (expect LEAKAGE) + scrubbed (expect CLEAN) ---
    # PoC uses the explicit offline-smoke key (auto would raise, by design — real KBs need CLAP/omni-embed).
    m_raw = kb_build.build_source(
        "heysquad_poc", "heysquad", None, records,
        key_modality="audio", value_type="text-fact", embedder="logmel-stats",
        audit_golds=golds, scrub=False, note="PoC raw (logmel-stats smoke key)",
    )
    m_clean = kb_build.build_source(
        "heysquad_poc_clean", "heysquad", None, records,
        key_modality="audio", value_type="text-fact", embedder="logmel-stats",
        audit_golds=golds, scrub=True, note="PoC answer-scrubbed (logmel-stats smoke key)",
    )

    # --- audio-query retrieval: self-match rate on a seeded eval slice ---
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(records))[:N_EVAL]
    eval_recs = [records[int(i)] for i in order]
    eval_ids = [r["from_item_id"] for r in eval_recs]
    src = kb_retrieve.load_source("heysquad_poc")
    hits = kb_retrieve.retrieve(src, [r["key_audio_ref"] for r in eval_recs], topk=5)
    self_at1 = float(np.mean([
        int(hits[i][0]["value"].provenance.get("from_item_id") == eval_ids[i])
        for i in range(len(eval_recs))
    ]))
    self_at5 = float(np.mean([
        int(any(h["value"].provenance.get("from_item_id") == eval_ids[i] for h in hits[i]))
        for i in range(len(eval_recs))
    ]))

    # --- snapshot freeze + replay ---
    snap_path = kb_snapshot.freeze_snapshot(
        "heysquad_poc", "heysquad", None, SEED, eval_ids, kb_build_hash=m_raw["build_hash"],
    )
    replay_order = np.random.default_rng(SEED).permutation(len(records))[:N_EVAL]
    replay_ids = [records[int(i)]["from_item_id"] for i in replay_order]
    replay = kb_snapshot.replay_matches("heysquad_poc", replay_ids)

    report = {
        "n_records": len(records),
        "index_backend": m_raw["index_backend"],
        "embedder": m_raw["embedder"],
        "dim": m_raw["dim"],
        "audit_raw_verdict": m_raw["leakage_audit"].get("verdict"),
        "audit_raw_overlap": m_raw["leakage_audit"].get("answer_overlap_rate"),
        "audit_clean_verdict": m_clean["leakage_audit"].get("post_scrub", {}).get("verdict"),
        "audit_clean_overlap": m_clean["leakage_audit"].get("post_scrub", {}).get("answer_overlap_rate"),
        "self_retrieval_at1": self_at1,
        "self_retrieval_at5": self_at5,
        "snapshot": snap_path,
        "replay_ok": replay["ok"],
    }
    checks = {
        "audit_flags_leakage": report["audit_raw_verdict"] == "LEAKAGE",
        "scrub_makes_clean": report["audit_clean_verdict"] == "CLEAN",
        "audio_self_match": self_at1 >= 0.95,
        "snapshot_replay": replay["ok"],
    }
    report["PASS"] = all(checks.values())
    report["checks"] = checks
    print("\n=== KB POC ===\n" + json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("KB_POC_PASS" if report["PASS"] else "KB_POC_FAIL", flush=True)
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
