"""T6 Step 1 — UNIFIED COMPRESSION-KEY feasibility (the memory system's first strategy).

Owner directive: "先探索统一编码的可行性，如果不行再按照不同任务编码" + "一定需要压缩后再进行使用".
Question (Stage-1 directional): can ONE unified, COMPRESSED speech key support task-relevant retrieval
across tasks? If yes, a single memory index is feasible; if no, fall back to per-task encoding.

Boundary (Information-Boundary-Guard): the key = the MODEL's OWN short content-summary of the audio,
embedded. It is deployable (the model can always summarize the test audio), reproducible at test time,
and used ONLY as a retrieval index — never injected as a golden answer. (Contrast the retracted M3, which
injected the human GOLDEN transcript as evidence.) No test-item label is ever used to build a key.

Metrics:
  (1) within-task retrieval precision@k on minds14 intents  vs chance  vs a RANDOM-key control (floor)
  (2) unified index: task-purity of neighbors (does the space separate tasks?)  +
      unified-vs-per-task precision (does pooling SQuAD-zh items hurt minds14 retrieval?)

reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/t6_compression_feasibility.py
"""
import json, os, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

BASE = p2.BASE
N_SLU = int(os.environ.get("T6_NSLU", "90"))
N_QA = int(os.environ.get("T6_NQA", "60"))
K = int(os.environ.get("T6_K", "5"))
SEED = 20260705
OUT = p2.DATA / "_repro" / "t6_compression_feasibility.json"
COMPRESS = ("In ONE short sentence, state the content of this audio — what is said or asked. "
            "Be concise; do not answer any question, just describe the content.")


def compress(wav):
    """Model-produced compression of the audio -> short text (the unified key source)."""
    content = [{"type": "input_audio", "input_audio": {"data": p2_b64(wav), "format": "wav"}},
               {"type": "text", "text": COMPRESS}]
    payload = {"messages": [{"role": "user", "content": content}], "max_tokens": 48,
               "seed": 42, "temperature": 0.0}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as z:
        return json.loads(z.read().decode())["choices"][0]["message"]["content"].strip()


def p2_b64(p):
    import base64
    return base64.b64encode(open(p, "rb").read()).decode()


def make_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")  # avoid GPU contention w/ llama-server
        return ("neural-minilm-multilingual", lambda texts: np.asarray(
            m.encode(list(texts), normalize_embeddings=True), dtype=np.float32))
    except Exception as e:
        print(f"  [embedder] neural unavailable ({type(e).__name__}); TF-IDF char-ngram fallback", flush=True)
        from sklearn.feature_extraction.text import TfidfVectorizer

        def emb(texts):
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            X = v.fit_transform(list(texts)).toarray().astype(np.float32)
            return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        return ("tfidf-char2-4", emb)


def knn_precision(keys, labels, pool_mask, query_mask, k):
    """For each query item, mean fraction of its k nearest (in `pool`, excluding self) sharing its label."""
    S = keys @ keys.T
    np.fill_diagonal(S, -1e9)
    S = S.copy()
    S[:, ~pool_mask] = -1e9           # restrict candidates to the pool
    precs, purity = [], []
    qidx = np.where(query_mask)[0]
    for i in qidx:
        nn = np.argsort(-S[i])[:k]
        precs.append(float(np.mean([labels[i] == labels[j] for j in nn])))
        purity.append(float(np.mean([task_of[i] == task_of[j] for j in nn])))
    return float(np.mean(precs)), float(np.mean(purity))


def chance_same_label(labels, mask):
    lab = labels[mask]
    _, cnt = np.unique(lab, return_counts=True)
    p = cnt / cnt.sum()
    return float(np.sum(p * p))       # expected same-label rate under random retrieval


task_of = None


def main():
    global task_of
    rng = np.random.default_rng(SEED)
    print(f"server={BASE} n_slu={N_SLU} n_qa={N_QA} k={K}", flush=True)
    p2.N_UTTS = N_SLU
    slu = p2.load_minds14_zh(np.random.default_rng(SEED))
    p2.N_UTTS = N_QA
    qa = p2.load_squad_zh(np.random.default_rng(SEED + 1))
    items = [dict(x, _task="slu", _lab=x["gold"]) for x in slu] + \
            [dict(x, _task="qa", _lab=-1) for x in qa]   # qa has no shared label taxonomy -> only for pooling
    print(f"  loaded slu={len(slu)} qa={len(qa)}", flush=True)

    cache = OUT.parent / "t6_comps_cache.json"
    t0 = time.time()
    if cache.exists():
        comps = json.load(open(cache))
        if len(comps) == len(items):
            print(f"  loaded {len(comps)} cached compressions", flush=True)
        else:
            comps = None
    else:
        comps = None
    if comps is None:
        comps = []
        for i, it in enumerate(items):
            try:
                comps.append(compress(it["wav"]))
            except Exception:
                comps.append("")
            if (i + 1) % 30 == 0:
                print(f"  compressed {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(comps, open(cache, "w"), ensure_ascii=False)

    emb_name, embed = make_embedder()
    keys = embed(comps)
    labels = np.array([it["_lab"] for it in items])
    task_of = np.array([it["_task"] for it in items])
    slu_mask = task_of == "slu"
    all_mask = np.ones(len(items), bool)
    comp_lens = [len(c) for c in comps]

    # (1) within-task (pool = slu only) precision + random-key floor
    p_pertask, _ = knn_precision(keys, labels, slu_mask, slu_mask, K)
    rng2 = np.random.default_rng(7)
    rand_keys = rng2.standard_normal(keys.shape).astype(np.float32)
    rand_keys /= np.clip(np.linalg.norm(rand_keys, axis=1, keepdims=True), 1e-9, None)
    p_random, _ = knn_precision(rand_keys, labels, slu_mask, slu_mask, K)
    chance = chance_same_label(labels, slu_mask)

    # (2) unified index (pool = all tasks): task purity + precision under pooling
    p_unified, purity_unified = knn_precision(keys, labels, all_mask, slu_mask, K)

    res = {
        "stage": "1-directional", "embedder": emb_name, "k": K,
        "n_slu": int(slu_mask.sum()), "n_qa": int((~slu_mask).sum()),
        "compression_chars_mean": round(float(np.mean(comp_lens)), 1),
        "within_task_precision@k": round(p_pertask, 3),
        "random_key_floor": round(p_random, 3),
        "chance_same_intent": round(chance, 3),
        "lift_over_chance": round(p_pertask - chance, 3),
        "unified_index_precision@k": round(p_unified, 3),
        "unified_task_purity@k": round(purity_unified, 3),
        "unified_vs_pertask_delta": round(p_unified - p_pertask, 3),
        "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/t6_compression_feasibility.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== T6 STEP1 COMPRESSION-KEY FEASIBILITY ===\n" + json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    print("T6_DONE", flush=True)


if __name__ == "__main__":
    main()
