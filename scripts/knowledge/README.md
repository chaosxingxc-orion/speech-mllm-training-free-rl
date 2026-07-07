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

## Reproducibility contract (Stage-1 floor)

Every experiment MUST freeze a `sample_manifest.json` (explicit `item_ids` + `slice_seed` + dataset
`revision` + `kb_build_hash`). `replay_matches` proves a fresh sampling reproduces the frozen slice
item-for-item, independent of parquet/row drift.

## Information-Boundary Guard (machine guardrail)

No source is admissible for a knowledge-**utilization** claim until its `leakage_audit.verdict ==
"CLEAN"`. `audit_texts` flags any source whose **values** contain the eval golds (HeySQuAD is `LEAKAGE`
by construction); `scrub_golds` produces the boundary-clean variant. The owner's recurring over-reach
failure mode becomes an automatic gate, not human discipline.

## Embedder tiers

`omni-embed-nemotron-3b` is the real audio key embedder (run in WSL/GPU) — the Stage-2 key. On a box
without GPU/network the build falls back to `logmel-stats-64` (mean+std of a log-mel spectrogram),
loudly flagged and **PoC-only** — enough to exercise the pipeline anywhere, not a Stage-2-grade key.
Index scaling: `FAISS-flat-IP` is exact cosine for Stage-1 sizes; swap to IVF/HNSW for large corpora.
