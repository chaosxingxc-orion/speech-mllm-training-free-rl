"""T2 (E7′) — PROPER task-definition few-shot: does teaching the task-HANDLING PATTERN activate the model?

The corrected ICL test (the old E7 only gave answers; owner: few-shot = 语音+文本任务定义样本, teach how a
CLASS of task is handled). Conditions, greedy, with paired-bootstrap CIs, all boundary-respecting
(demos from TRAIN, test item audio-only, no leakage):
  A  plain instruction (= P2 baseline)
  B  + TASK-DEFINITION (how-to-handle text, 0-shot)         -> does the METHOD instruction alone help?
  C  + task-def + k reasoned demos (audio, reasoning, answer) from train  -> full proper few-shot
  b1 + SHUFFLED task-def + k demos                          -> does the task-def CONTENT matter (b2) vs format (b1)?

reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/p4_taskdef_fewshot.py --only mmau-mini,vocalbench-zh
"""
import base64, json, os, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

BASE = p2.BASE
K = int(os.environ.get("T2_K", "2"))
K_POOL = int(os.environ.get("T2_POOL", "8"))
N_QUERY = int(os.environ.get("T2_NQUERY", "50"))
SEED = 20260705
NBOOT = 10000
OUT = p2.DATA / "_repro" / "t2_taskdef_fewshot.json"

TASKDEF = {
    "mcq": ("Approach for this task: listen to the ENTIRE audio, identify the specific acoustic evidence "
            "the question asks about, rule out options that contradict that evidence, then choose the single "
            "best-supported option."),
    "qa": ("Approach for this task: first determine exactly what the spoken question is asking, then recall "
           "or infer the specific answer, then respond with only the short answer."),
    "slu": ("Approach for this task: identify the caller's underlying banking intent from what they say, and "
            "map it to the closest intent category."),
}
SHUF = ("Approach for this task: ignore the audio content, focus on the background noise texture, and answer "
        "based on ambient acoustics.")  # deliberately wrong handling-pattern = b1 floor


def _b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def call(content, temp=0.0, seed=42, maxtok=None):
    payload = {"messages": [{"role": "user", "content": content}],
               "max_tokens": maxtok or p2.MAXTOK, "seed": seed, "temperature": temp}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as z:
        return json.loads(z.read().decode())["choices"][0]["message"]["content"].strip()


def audio_part(wav):
    return {"type": "input_audio", "input_audio": {"data": _b64(wav), "format": "wav"}}


def demo_reasoning(item):
    """Generate a brief handling-reasoning for a TRAIN demo (label available at train time — legitimate)."""
    ans = p2.LETTERS[item["gold"]] + ". " + item["opts"][item["gold"]] if item["task"] == "mcq" else \
        (f"{item['gold']}. {p2.MINDS14[item['gold']]}" if item["task"] == "slu" else str(item["gold"]))
    c = [audio_part(item["wav"]),
         {"type": "text", "text": f"{item['instr']}\nThe correct answer is: {ans}\n"
          f"In ONE sentence, state the reasoning that leads to it (the handling pattern for this task)."}]
    try:
        rsn = call(c, maxtok=60)
    except Exception:
        rsn = ""
    return {"wav": item["wav"], "reasoning": rsn, "answer": ans, "instr": item["instr"]}


def build_content(query, taskdef, demos):
    content = []
    if taskdef:
        content.append({"type": "text", "text": taskdef})
    for d in demos:
        content.append(audio_part(d["wav"]))
        content.append({"type": "text", "text": f"{d['instr']}\nReasoning: {d['reasoning']}\nAnswer: {d['answer']}"})
    content.append(audio_part(query["wav"]))
    content.append({"type": "text", "text": query["instr"]})
    return content


def boot(d, seed=1):
    rng = np.random.default_rng(seed)
    return [round(float(np.quantile([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(NBOOT)], q)), 4)
            for q in (.025, .975)]


def run(name, loader):
    rng = np.random.default_rng(SEED)
    p2.N_UTTS = K_POOL + N_QUERY + 20
    items = loader(rng)
    if len(items) < K_POOL + 10:
        return {"error": "too few"}
    pool, query = items[:K_POOL], items[K_POOL:K_POOL + N_QUERY]
    task = pool[0]["task"]
    td = TASKDEF.get(task, TASKDEF["qa"])
    demos_all = [demo_reasoning(x) for x in pool]   # generate reasoning once, reuse
    t0 = time.time()
    A, B, C, b1 = [], [], [], []
    for qi, q in enumerate(query):
        sel = [demos_all[(qi + j) % K_POOL] for j in range(K)]
        try:
            A.append(p2.reward(q, call(build_content(q, None, []))))
            B.append(p2.reward(q, call(build_content(q, td, []))))
            C.append(p2.reward(q, call(build_content(q, td, sel))))
            b1.append(p2.reward(q, call(build_content(q, SHUF, sel))))
        except Exception:
            continue
        if (qi + 1) % 25 == 0:
            print(f"  [{name} {qi+1}] A={np.mean(A):.3f} B={np.mean(B):.3f} C={np.mean(C):.3f} b1={np.mean(b1):.3f} ({time.time()-t0:.0f}s)", flush=True)
    A, B, C, b1 = map(np.array, (A, B, C, b1))
    res = {"task": task, "n": len(A), "K": K,
           "A_plain": round(float(A.mean()), 4), "B_taskdef_only": round(float(B.mean()), 4),
           "C_taskdef_fewshot": round(float(C.mean()), 4), "b1_shuffled_taskdef": round(float(b1.mean()), 4),
           "C_minus_A": round(float((C - A).mean()), 4), "C_vs_A_CI": boot(C - A),
           "B_minus_A": round(float((B - A).mean()), 4), "B_vs_A_CI": boot(B - A),
           "C_minus_b1_b2genuine": round(float((C - b1).mean()), 4), "C_vs_b1_CI": boot(C - b1),
           "C_rel_gain": round(float((C.mean() - A.mean()) / max(1e-9, A.mean())), 4)}
    print(f"  == {name}: A={res['A_plain']} B={res['B_taskdef_only']} C={res['C_taskdef_fewshot']} "
          f"b1={res['b1_shuffled_taskdef']} | C-A={res['C_minus_A']:+.3f} CI{res['C_vs_A_CI']} "
          f"C-b1={res['C_minus_b1_b2genuine']:+.3f} CI{res['C_vs_b1_CI']} rel={res['C_rel_gain']:+.1%}", flush=True)
    return res


def main():
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    print(f"server={BASE} K={K} pool={K_POOL} nquery={N_QUERY}", flush=True)
    out = json.load(open(OUT)).get("results", {}) if OUT.exists() else {}
    for name, loader in p2.LOADERS.items():
        if only and name not in only:
            continue
        try:
            out[name] = run(name, loader)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "lever": "PROPER task-definition few-shot (E7′)",
                               "conditions": "A plain / B taskdef-only / C taskdef+reasoned-demos / b1 shuffled-taskdef",
                               "K": K, "n_query": N_QUERY, "seed": SEED,
                               "boundary": "demos from train, test audio-only, no leakage"},
                   "results": out, "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/p4_taskdef_fewshot.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== T2 TASKDEF FEW-SHOT ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("T2_DONE", flush=True)


if __name__ == "__main__":
    main()
