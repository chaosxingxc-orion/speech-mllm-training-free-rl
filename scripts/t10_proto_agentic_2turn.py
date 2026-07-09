"""E4 (proto-agentic) — minimal 2-turn TOOL-style knowledge injection, vs single-turn injection.

Full agentic knowledge benchmarks (eva-bench simulator / tau2 DB-env / audiomc o4-mini rubric) are
INFEASIBLE offline. This is the honest minimal agentic touch: a 2-turn tool-call pattern
  turn1  user: [audio question]
  turn2  assistant: "[检索请求]"  ->  user: "检索结果: {knowledge}"  ->  answer
compared to single-turn injection of the SAME (counterfactual) knowledge. Tests whether the agentic
(multi-turn tool) framing changes knowledge UTILISATION (follow-rate) vs a flat single-turn prompt.
Boundary-clean: injected value A' is a controlled counterfactual (another item's gold), never this
item's gold; audio input; we score adoption of the injected value.

reproduce: SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python -u scripts/t10_proto_agentic_2turn.py
"""
import base64, json, os, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

N = int(os.environ.get("T10_N", "40"))
SEED = 20260707
OUT = p2.DATA / "_repro" / "t10_proto_agentic_2turn.json"
ANS = "请只用一个简短答案回答音频里的问题(只给答案,不要解释)。"
CF1 = "参考资料:根据资料,该问题的正确答案是「{a}」。\n\n"


def _b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def gen_msgs(messages):
    payload = {"messages": messages, "max_tokens": 64, "seed": 42, "temperature": 0.0}
    req = urllib.request.Request(p2.BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def single_turn(wav, a):
    return gen_msgs([{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": _b64(wav), "format": "wav"}},
        {"type": "text", "text": CF1.format(a=a) + ANS}]}])


def two_turn(wav, a):
    return gen_msgs([
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": _b64(wav), "format": "wav"}},
            {"type": "text", "text": "回答音频里的问题;如需外部资料可先请求检索。"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "[检索请求] 查询该问题的相关资料。"}]},
        {"role": "user", "content": [{"type": "text", "text": f"检索结果:该问题的正确答案是「{a}」。\n\n{ANS}"}]},
    ])


def main():
    rng = np.random.default_rng(SEED)
    p2.N_UTTS = N
    items = [it for it in p2.load_squad_zh(rng) if it["task"] == "qa" and str(it["gold"]).strip()]
    n = len(items)
    golds = [str(it["gold"]) for it in items]
    aprime = [golds[(k + 1) % n] for k in range(n)]
    st_follow, tt_follow = [], []
    t0 = time.time()
    for k, it in enumerate(items):
        A, Ap = golds[k], aprime[k]
        if p2.norm(Ap) == p2.norm(A):
            continue
        try:
            os1 = single_turn(it["wav"], Ap)
            ot = two_turn(it["wav"], Ap)
        except Exception as e:
            print(f"  skip #{k}: {type(e).__name__}", flush=True); continue
        st_follow.append(int(p2.norm(Ap) in p2.norm(os1)))
        tt_follow.append(int(p2.norm(Ap) in p2.norm(ot)))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{n}] single_follow={np.mean(st_follow):.3f} twoturn_follow={np.mean(tt_follow):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    def m(x):
        return round(float(np.mean(x)), 3) if x else None
    out = {"stage": "1-directional", "grade": "small-n, proto-agentic (2-turn tool-inject), boundary-clean",
           "model": "qwen3-omni-30b Q8_0 GGUF", "dataset": "SQuAD-zh", "n": len(st_follow),
           "single_turn_CF_follow": m(st_follow), "two_turn_tool_CF_follow": m(tt_follow),
           "delta_twoturn_minus_single": (round(m(tt_follow) - m(st_follow), 3) if st_follow else None),
           "note": "Full agentic KB benchmarks (eva/tau2/audiomc) infeasible offline (simulator/DB-env/o4-mini "
                   "rubric); this is the minimal agentic touch — does 2-turn tool-style injection change "
                   "utilisation vs single-turn. Boundary-clean (A'=counterfactual, audio input).",
           "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python -u scripts/t10_proto_agentic_2turn.py"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== T10 PROTO-AGENTIC ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("wrote", OUT, "\nT10_DONE", flush=True)


if __name__ == "__main__":
    main()
