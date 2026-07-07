# Design rationale — why speech-keyed KV is the right (mainstream) architecture

Owner asked (2026-07-07): compare against mainstream industry practice; confirm that
**KEY = speech-embedding matrix + persistent vector index, VALUE = another modality (transcript /
labels / …) stored separately** is the reasonable mode — and that the old TF-IDF text-passage pool
was "last-century tech." This doc records the verified comparison. Verdict: **YES — this is the
standard vector-retrieval shape; the rejected design is genuinely outdated.**

## Verdict in one line

The design is **not a "prove it's reasonable" question — it IS the standard form** of mainstream
vector retrieval. `{vector = key, payload = value, ANN search}` is exactly the data model of every
major vector DB; **Qdrant/Milvus's `{id, vector, payload}` is a ready-made implementation of it.**
Audio-as-key is backed by Shazam fingerprinting (industrial), CLAP/CLIP cross-modal retrieval,
speaker-embedding DBs, and **WavRAG (ACL 2025)**.

## 1. Vector-DB architecture — all mainstream systems use embedding-key + payload-value + ANN

| system | role | mainstream index | key/value model | persisted |
|---|---|---|---|---|
| **FAISS** (Meta) | library/algorithm layer | Flat, IVF, IVF-PQ, HNSW, PQ | index stores the **vector matrix**, returns int ids; id→value is user-maintained | self-built |
| **Milvus/Zilliz** | billion-scale distributed | IVF·, HNSW, DiskANN, PQ/SQ | collection = vector field + scalar (payload) fields | yes |
| **Qdrant** | payload-filtering-first | **HNSW** + quantization | **Point = {id, vector, payload}** (docs: vector=semantics, payload=metadata) | yes |
| **Pinecone** | serverless managed | managed graph + sharding | vector + metadata upsert | yes (managed) |
| **Weaviate** | semantic + hybrid + schema | **HNSW** | object = vector + properties | yes |
| **Chroma** | lightweight/local RAG | HNSW | embedding + documents + metadata | yes |
| **pgvector / OpenSearch** | vectors inside existing DB | HNSW (pgvector also IVFFlat) | vector column + normal rows | yes |

**HNSW is the de-facto default** (Qdrant/Weaviate/pg/ES). **IVF-PQ / DiskANN** for out-of-RAM
billion-scale. **Flat** only for small/exact. Our `kb_index` uses FAISS-flat-IP for Stage-1 sizes and
now auto-switches to **HNSW** at n≥50k; billion-scale → IVF-PQ or a managed store.

## 2. Embedding-key + value separation IS the RAG de-facto standard

Every major store separates "the vector you search on" from "the payload you retrieve" — the exact
skeleton of this design. Using **dense KV in place of a TF-IDF text pool is a forward upgrade**
(semantic generalization + persistence + filtering + scalability). Sparse retrieval (BM25) has NOT
vanished — it remains a strong **hybrid (dense+sparse)** lane (BEIR/MS-MARCO; ~5–30% NDCG gains on
exact-term / proper-noun / ID matching), but as an *optional add-on route*, not a return to TF-IDF.

## 3. Audio-as-key cross-modal KV retrieval — mature, multiple precedents

- **CLAP** (Microsoft): dual-encoder InfoNCE embeds audio + text into one shared space → audio
  embeddings are natural retrieval keys.
- **Audio fingerprinting (Shazam / ACRCloud)**: audio → hash **key** → {track id, offset} **value**;
  the oldest industrial "audio-as-key → retrieve value" system (structurally identical, LSH not dense).
- **WavRAG (ACL 2025, Zhao group)**: first **audio-native RAG** — bypasses ASR, embeds raw audio,
  cosine-retrieves an **offline pre-encoded** knowledge base; ~**10× faster** than an ASR→text-RAG
  pipeline at comparable retrieval. Directly endorses "audio-centric > transcribe-then-retrieve."
- **CLIP image→caption via FAISS** and **speaker-embedding DBs (x-vector/ECAPA)**: the same
  embedding→ANN→retrieve-payload recipe in vision and speaker-ID.

> One nuance: WavRAG builds a **symmetric** unified audio-text space (query/doc any modality); ours is
> **asymmetric** (audio-only key, value stored raw). Both are mainstream — asymmetric = Shazam /
> CLIP-retrieval / speaker-DB pattern (simpler, enough for "audio query → transcript/label"); symmetric
> = more flexible if text or mixed queries are later needed. Choose by use-case.

## 4. Improvement roadmap (engineering choices, not route corrections)

1. **Index selection**: HNSW default for ≤ hundreds-of-millions, dynamic add/delete, high recall;
   IVF-PQ / DiskANN for out-of-RAM. (Implemented: `VectorIndex.build(index_type=...)`.)
2. **Normalize + cosine**: L2-normalize keys & queries, inner-product = cosine. (Implemented.)
3. **Payload filtering**: carry speaker/language/time/domain as payload fields → filter-then-search.
   (Schema carries `provenance`; a managed store would give native filtering.)
4. **Optional hybrid (dense+sparse)**: BM25/sparse lane + RRF for exact-term value-side matches.
5. **Embedder benchmark**: A/B `omni-embed-nemotron-3b` vs CLAP by retrieval recall@k — don't default.
6. **Symmetric vs asymmetric**: revisit if text/mixed queries are needed (WavRAG-style shared space).
7. **Incremental update + snapshot consistency**: upsert/delete; persist index+payload+id-map together
   (the core value a vector DB adds over raw FAISS).
8. **Offline pre-encoding**: batch-encode keys at build time (WavRAG convention) — our persisted index
   already implies this; make it an explicit build stage.

## Productionization note

For Stage-2 scale, consider **Qdrant or Milvus** instead of raw FAISS: their `{id, vector, payload}`
model provides value-separation, metadata filtering, persistence, and incremental update out of the
box — exactly this design, managed. Our `kb_*` layer keeps the same conceptual contract so a swap is
mechanical.

## Sources
FAISS/Milvus/Qdrant/Pinecone/Weaviate/Chroma comparisons (2025–26); Qdrant Points/Payload docs;
HNSW/IVF-PQ/DiskANN index guides; dense-sparse hybrid retrieval (BEIR/MS-MARCO); WavRAG
(aclanthology 2025.acl-long.613 / arXiv 2502.14727); CLAP (microsoft/CLAP); Shazam/ACRCloud audio
fingerprinting (+ arXiv 2410.21876); CLIP→FAISS retrieval (arXiv 2401.06167); x-vectors speaker
embeddings. Full URLs archived in the WS-B build note / Decision-Log.
