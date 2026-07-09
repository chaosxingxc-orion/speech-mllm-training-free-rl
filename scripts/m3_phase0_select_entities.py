"""M3 Phase-0 entity/utterance selection (CPU; deterministic; refs-only — no model behavior consulted).

Pre-registered under wiki/2026-07-03-agentic-tfrl-step1-preregistration.md (freeze b19bff2) §5 M3,
implemented per the protocol spec's §B.2/§B.4 (architect-hardened; interpretation log §D):
  - recurring rare entity = (speaker-chapter, entity) with entity a pure-alphabetic NORM token
    (len>=4), train-960h count <= 5 (rarity vs the model-scale read-speech distribution),
    recurring in >= 3 utterances of ONE speaker-chapter, and <= 1 utterance elsewhere corpus-wide;
  - candidates sorted (train_freq asc, chapter asc, entity asc) — rarest first, no RNG;
  - up to 3 utts per pair, dev-spent 144 C1 ids EXCLUDED, one target entity per utterance
    (rarest claims it), cap 36 utterances;
  - expected on today's on-disk data: 26 candidate pairs / 36 utts / 13 entities / 9 chapters,
    all selected entities train-count 0 — any deviation is a selection bug (hard assert).

Also builds/caches the train-960h token-frequency table (shared with the M5 pilot) and writes the
clean wavs. Commit the emitted selection artifact BEFORE any GPU sample exists (pre-commitment).

reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/m3_phase0_select_entities.py
"""
import collections, glob, io, json, os, re, time
from pathlib import Path

DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
ART = Path(__file__).resolve().parent.parent / "_repro" / "asr_bon_llamacpp_snr5.json"
FREQ_CACHE = DATA / "_repro" / "librispeech_train960_tokfreq.json"
OUT = DATA / "_repro" / "m3_phase0_selection.json"
WAV_DIR = DATA / "_repro" / "m3_phase0_wavs"
TRAIN_MAX = 5
K_MIN = 3
N_UTTS = 36
MAX_PER_PAIR = 3
TOK_RE = re.compile(r"^[A-Z]{4,}$")
EXPECT = {"pairs": 26, "utts": 36, "entities": 13, "chapters": 9}


def train_freq_table() -> dict:
    if FREQ_CACHE.exists():
        return json.load(open(FREQ_CACHE))
    import pyarrow.parquet as pq
    freq = collections.Counter()
    n_tok = 0
    for split in ("train.clean.100", "train.clean.360", "train.other.500"):
        for f in sorted(glob.glob(str(DATA / "datasets/librispeech/all" / split / "*.parquet"))):
            for s in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                w = s.split()
                freq.update(w)
                n_tok += len(w)
    print(f"train tokens: {n_tok}  vocab: {len(freq)}", flush=True)
    FREQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(freq, open(FREQ_CACHE, "w"))
    return dict(freq)


def main():
    t0 = time.time()
    import pyarrow.parquet as pq
    import soundfile as sf
    freq = train_freq_table()
    t = pq.read_table(DATA / "datasets/librispeech/all/test.other/0000.parquet",
                      columns=["audio", "text", "id"])
    ids = t.column("id").to_pylist()
    text = t.column("text").to_pylist()
    dev_spent = set(r["id"] for r in json.load(open(ART))["per_utt"])

    utts_by_chap = collections.defaultdict(list)
    for i, s in zip(ids, text):
        utts_by_chap["-".join(i.split("-")[:2])].append((i, s))
    df_corpus = collections.Counter()
    df_chap = {}
    for ch, utts in utts_by_chap.items():
        c = collections.Counter()
        for _, s in utts:
            for w in set(s.split()):
                c[w] += 1
                df_corpus[w] += 1
        df_chap[ch] = c

    candidates = []  # (train_freq, chapter, entity, n_in_chapter)
    for ch, c in df_chap.items():
        for w, n_in in c.items():
            if n_in < K_MIN or not TOK_RE.match(w):
                continue
            if freq.get(w, 0) > TRAIN_MAX:
                continue
            if df_corpus[w] - n_in > 1:
                continue  # chapter-exclusive (<=1 stray utterance elsewhere)
            candidates.append((freq.get(w, 0), ch, w, n_in))
    candidates.sort()

    target = {}
    for tf, ch, w, n_in in candidates:
        for u in [i for i, s in utts_by_chap[ch] if w in s.split() and i not in dev_spent][:MAX_PER_PAIR]:
            if u not in target:
                target[u] = {"entity": w, "train_freq": tf, "chapter": ch, "recurrence": n_in}
        if len(target) >= N_UTTS:
            break
    sel = dict(sorted(target.items())[:N_UTTS])
    ref = dict(zip(ids, text))
    ents = set(v["entity"] for v in sel.values())
    chaps = set(v["chapter"] for v in sel.values())
    got = {"pairs": len(candidates), "utts": len(sel), "entities": len(ents), "chapters": len(chaps)}
    print("selection:", got, flush=True)
    assert got == EXPECT, f"selection deviates from the architect-verified dry-run targets: {got} != {EXPECT}"
    assert all(v["train_freq"] == 0 for v in sel.values()), "expected all selected entities train-OOV"

    # clean wavs (no noise — kill-biased condition, interpretation log #3), single audio-column pass
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    need = [u for u in sel if not (WAV_DIR / f"{u}.wav").exists()]
    if need:
        idx = {u: i for i, u in enumerate(ids)}
        audio = t.column("audio").to_pylist()
        for u in need:
            wav, sr = sf.read(io.BytesIO(audio[idx[u]]["bytes"]), dtype="float32")
            sf.write(str(WAV_DIR / f"{u}.wav"), wav, sr)
    print(f"wavs ready: {len(sel)} in {WAV_DIR}", flush=True)

    out = {
        "summary": {"lane": "M3", "phase": 0, "n_candidate_pairs": len(candidates),
                    "n_utts": len(sel), "n_entities": len(ents), "n_chapters": len(chaps),
                    "selection_rule": {"rarity": f"librispeech-train960 count <= {TRAIN_MAX}",
                                       "k_min_recurrence": K_MIN, "chapter_exclusive": True,
                                       "min_len": 4, "dev_spent_excluded": len(dev_spent),
                                       "max_utts_per_pair": MAX_PER_PAIR, "deterministic": "no RNG"},
                    "condition": "clean",
                    "prereg": "wiki/2026-07-03-agentic-tfrl-step1-preregistration.md @ b19bff2 §5 M3"},
        "candidate_pairs": [{"train_freq": tf, "chapter": ch, "entity": w, "n_in_chapter": n}
                             for tf, ch, w, n in candidates],
        "selected": [{"id": u, **v, "ref": ref[u]} for u, v in sel.items()],
        "elapsed_s": round(time.time() - t0, 1),
        "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/m3_phase0_select_entities.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    print("M3_SELECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
