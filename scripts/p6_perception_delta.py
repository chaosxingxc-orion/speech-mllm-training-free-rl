"""EXP (perception-delta) — is the omni's ">transcript delta" load-bearing? (D0 core research point)

Boundary-clean (Information-Boundary-Guard): the transcript is the MODEL's OWN ASR output — deployable,
reproducible at test, NOT a golden reference. So this is exactly the deployment question "can I replace the
frozen omni with ASR->text-LLM (using the omni's own ASR)?". Same frozen model in both arms:
  A  omni(audio)                      — full perception
  B  omni(own-transcript, text-only)  — audio discarded, only the model's ASR text remains (ASR->LLM proxy)
delta = acc(A) - acc(B) = what the audio carries BEYOND its transcript. Expect: large on audio-dependent
tasks (mmau: sounds/music/events), ~0 on purely-lexical spoken-QA (SQuAD-zh). If delta>0 the omni is NOT
replaceable by ASR->text-LLM -> the perception delta is real and load-bearing -> omni is a genuine
new-info (perception) element.

reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/p6_perception_delta.py --only mmau-mini,SQuAD-zh,vocalbench-zh
"""
import base64, json, os, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

BASE = p2.BASE
N = int(os.environ.get("P6_N", "60"))
SEED = 20260706
NBOOT = 10000
OUT = p2.DATA / "_repro" / "p6_perception_delta.json"
TRANSCRIBE = ("Transcribe the spoken content of this audio verbatim, in its original language. "
              "If there is no intelligible speech, reply exactly: [no speech]")


def _b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def _post(content, maxtok=None):
    payload = {"messages": [{"role": "user", "content": content}],
               "max_tokens": maxtok or p2.MAXTOK, "seed": 42, "temperature": 0.0}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as z:
        return json.loads(z.read().decode())["choices"][0]["message"]["content"].strip()


def with_audio(wav, text, maxtok=None):
    return _post([{"type": "input_audio", "input_audio": {"data": _b64(wav), "format": "wav"}},
                  {"type": "text", "text": text}], maxtok)


def text_only(text, maxtok=None):
    return _post([{"type": "text", "text": text}], maxtok)


def boot(d):
    rng = np.random.default_rng(1)
    return [round(float(np.quantile([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(NBOOT)], q)), 4)
            for q in (.025, .975)]


def run(name, loader):
    p2.N_UTTS = N + 15
    items = loader(np.random.default_rng(SEED))[:N]
    if len(items) < 10:
        return {"error": "too few"}
    A, B, tlens = [], [], []
    t0 = time.time()
    for i, it in enumerate(items):
        try:
            a = with_audio(it["wav"], it["instr"])
            tr = with_audio(it["wav"], TRANSCRIBE, maxtok=200)
            b = text_only(f"Transcript of the audio:\n\"\"\"{tr}\"\"\"\n\n{it['instr']}")
        except Exception:
            continue
        A.append(p2.reward(it, a))
        B.append(p2.reward(it, b))
        tlens.append(len(tr))
        if (i + 1) % 20 == 0:
            print(f"  [{name} {i+1}] audio={np.mean(A):.3f} transcript={np.mean(B):.3f} ({time.time()-t0:.0f}s)", flush=True)
    A, B = np.array(A, float), np.array(B, float)
    d = A - B
    ci = boot(d)
    sig = "SIG(+)" if ci[0] > 0 else ("SIG(-)" if ci[1] < 0 else "n.s.")
    res = {"task": items[0]["task"], "n": len(A),
           "acc_audio": round(float(A.mean()), 4), "acc_transcript_only": round(float(B.mean()), 4),
           "perception_delta": round(float(d.mean()), 4), "delta_CI": ci, "sig": sig,
           "transcript_chars_mean": round(float(np.mean(tlens)), 1)}
    print(f"  == {name}: audio={res['acc_audio']} transcript={res['acc_transcript_only']} "
          f"delta={res['perception_delta']:+.3f} CI{ci} [{sig}]", flush=True)
    return res


def main():
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    print(f"server={BASE} n={N} seed={SEED}", flush=True)
    out = json.load(open(OUT)).get("results", {}) if OUT.exists() else {}
    for name, loader in p2.LOADERS.items():
        if only and name not in only:
            continue
        try:
            out[name] = run(name, loader)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "exp": "perception-delta: omni(audio) vs omni(own-transcript)",
                               "boundary": "transcript is model-produced (deployable), not golden; same frozen model both arms",
                               "n": N, "seed": SEED},
                   "results": out, "reproduce": "python scripts/p6_perception_delta.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== EXP PERCEPTION-DELTA ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("P6_DONE", flush=True)


if __name__ == "__main__":
    main()
