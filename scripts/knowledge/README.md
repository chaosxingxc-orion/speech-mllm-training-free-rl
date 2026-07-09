# speechrl knowledge base — the standard (SPEECH-KEYED)

Dataset-agnostic, **speech-keyed**, modality-aware, **persistent**, **auditable** knowledge base for a
frozen speech/omni model. This directory is the **standard operating layer**, co-located with the
knowledge under `SPEECHRL_KB_DIR` (default `E:\speechrl-knowledge`).

> Version-controlled source lives in the work repo at `scripts/knowledge/`. A deployed copy is
> mirrored here to `E:\speechrl-knowledge\ops\` by `deploy_to_e.py`, stamped with the git sha in
> `ops/VERSION`. Edit the repo copy; never hand-edit the E-drive mirror.

## The architecture (owner, 2026-07-07)

A **multimodal** knowledge base is keyed on **speech**, not text:

```
KEY   = a dense AUDIO embedding      -> a row in keys.npy, indexed by a persisted vector index
VALUE = another modality             -> transcript / labels / intent / answer / text-fact / other-modal
```

Retrieval = **embed query audio → ANN search over the key index → return the value**. This is the
mainstream vector-DB / RAG shape (embedding index + payload store, as in FAISS / Milvus / Qdrant /
Pinecone), applied with **audio as the query key** — the genuinely multimodal design. It replaces the
pre-2026-07-07 "KB": a **transient, in-RAM, TF-IDF text-passage pool** (`scripts/t7_rag_gate_probe.py`)
that was text-keyed, index-less, dataset-hardcoded, and had **no leakage check** — the exact "last
century" shape that produced the T7 answer-lookup leak. Text-keyed retrieval is retained only as a
**legacy** path for reading-comprehension passage pools.

## Layout (under `SPEECHRL_KB_DIR`)

```
knowledge_base/<source>/keys.npy       KEY matrix [N×d] — audio embeddings, L2-normalized
knowledge_base/<source>/index.faiss    persisted ANN index over the keys (FAISS-flat-IP; optional)
knowledge_base/<source>/values.jsonl   row-aligned VALUES (the payload; one KnowledgeValue per line)
knowledge_base/<source>/retriever.pkl  legacy: fitted TF-IDF vectorizer (only if key_modality=='text')
knowledge_base/<source>/manifest.json  SourceManifest: embedder + key_modality + dim + provenance + hash
snapshots/<experiment>/sample_manifest.json   frozen eval slice: item ids + seed + revision + kb hash
ops/                                    this operating layer (deployed copy of scripts/knowledge/)
```

## The standard operations

| op | module | what it does |
|----|--------|--------------|
| embed (key) | `kb_embed.embed_audio` | waveform → dense key vector (omni-embed-nemotron-3b; logmel-stats PoC fallback) |
| index | `kb_index.VectorIndex` | KEY matrix → persisted ANN index (FAISS-flat-IP else numpy-flat cosine) |
| build | `kb_build.build_source` | dataset → keys.npy + index + values.jsonl + manifest; runs value-side leakage audit |
| retrieve | `kb_retrieve.load_source` + `retrieve` | reload persisted source, embed query audio, ANN top-k → values |
| gate | `kb_retrieve.gate` | R1 admission gate — **study-only** (T7: precision-gating HURTS a strong omni) |
| audit | `kb_audit.audit_texts` / `audit_source` | value-side answer-overlap leakage check; `scrub_golds` → clean |
| inject | `kb_inject.deliver` | delivery FORM selection — `two_turn_tool` (t10) ~doubles adoption; the clean lever |
| snapshot | `kb_snapshot.freeze_snapshot` / `replay_matches` | freeze + verify a reproducible eval slice |
| registry | `kb_registry.REGISTRY` | full 28-dataset map: key_modality × value_type × status |
| gate test | `test_kb_gate.py` | offline proof of the enforcement gate (§ Information-Boundary Guard below) |

## Reproducibility contract (Stage-1 floor)

Every experiment MUST freeze a `sample_manifest.json` (explicit `item_ids` + `slice_seed` + dataset
`revision` + `kb_build_hash`). `replay_matches` proves a fresh sampling reproduces the frozen slice
item-for-item, independent of parquet/row drift.

## Information-Boundary Guard (machine guardrail)

No source is admissible for a knowledge-**utilization** claim until its `leakage_audit.verdict ==
"CLEAN"`. `audit_texts` flags any source whose **values** contain the eval golds (HeySQuAD is `LEAKAGE`
by construction); `scrub_golds` produces the boundary-clean variant.

This is enforced in **code**, at both ends of the persistence boundary, not left to human discipline:

- **Build-time gate** (`kb_build.build_source`): if the audit's final verdict (post-scrub if
  `scrub=True`, else raw) is `LEAKAGE`, the call **raises `kb_schema.KBLeakageError` and refuses to
  persist** the source. The escape hatch is `force_persist=True` (PoC/debug only) — it logs a loud
  `WARNING` and stamps `manifest.forced = True`, so a force-built leaking source can never be mistaken
  for an admissible one downstream.
- **Load-time gate** (`kb_retrieve.load_source`): reads `manifest.leakage_audit` and **raises
  `KBLeakageError` unless the final verdict is exactly `"CLEAN"`** — an unaudited source (`verdict is
  None`, i.e. built without `audit_golds`) is treated as NOT admissible either, since audit silence is
  not a clean bill. The escape hatch is `allow_unclean=True` (PoC/debug only) — it logs a loud
  `WARNING` on every load.

Both escape hatches exist only for pipeline-mechanics smoke tests (e.g. `kb_poc.py` deliberately
builds+loads a raw `LEAKAGE` source to prove the audit catches it) — never use them to back a
knowledge-utilization result. `scripts/knowledge/test_kb_gate.py` proves the gate: a `LEAKAGE` source
cannot be built without `force_persist=True`, cannot be loaded without `allow_unclean=True`, and a
`CLEAN` source passes both without any flag.

## Embedder tiers

`omni-embed-nemotron-3b` is the real audio key embedder — the Stage-2 key. Wired via its **official
asymmetric `sentence-transformers` API** (`kb_embed._omni_model`:
`SentenceTransformer(model_dir, trust_remote_code=True, device=...)` — `trust_remote_code=True` is
REQUIRED, the checkpoint ships custom `NVOmniEmbedModel` code): the document/KEY side uses
`encode_document([{'audio': path}, ...])` (`kb_embed._omni_embed`, called from `embed_audio`), and the
query side uses `encode_query([text, ...])` (`kb_embed._omni_embed_query`, reachable via
`embed_text(embedder='omni-embed')` for cross-modal text→audio-KB retrieval). **Device defaults to
CPU** — the GPU is held by the resident llama-server and must never be touched (CLAUDE.md); pass
`device='cuda'` explicitly only when the GPU is confirmed free.

A CPU load of this ~4.7B-param model is expected to be slow (multi-minute cold start, unverified in
this session — never run it inline in an automated/CI path without first timing it manually). It is
intentionally **not** exercised by any automated test in this directory (`test_kb_gate.py` / `kb_poc.py`
both use the offline `logmel-stats` fallback instead). To smoke-test the omni-embed path itself, run it
manually with a generous timeout, e.g.:

```bash
SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data \
  python -c "
import sys; sys.path.insert(0, 'scripts/knowledge')
import kb_embed
name, mat = kb_embed.embed_audio(['<path-to-a-wav>'], embedder='omni-embed')
print(name, mat.shape)
"
```

On a box without GPU/network (or before that manual smoke has been run), the build falls back to
`logmel-stats-64` (mean+std of a log-mel spectrogram), loudly flagged and **PoC-only** — enough to
exercise the pipeline anywhere, not a Stage-2-grade key. Index scaling: `FAISS-flat-IP` is exact cosine
for Stage-1 sizes; swap to IVF/HNSW for large corpora.
