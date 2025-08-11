# Multilingual Multi‑Hop RAG (ES/PT/EN)

This is my personal project to build a local‑first RAG system that can answer questions in Spanish, Portuguese, and English. It retrieves from a Postgres+pgvector database, embeds with **BGE‑M3** on my RTX 3060, and will eventually do multi‑hop reasoning.

---

## What I’ve got working so far

* **Chunking:** Took a slice of HotpotQA, split into `documents.jsonl` and `chunks.jsonl` with deduplication and word overlap.
* **Database:** Postgres running in Docker with **pgvector**.
* **Loading:** A script to create tables and bulk‑load the JSONL files.
* **Embedding:** Another script that runs **BAAI/bge‑m3** on GPU, normalizes the embeddings, and stores them in the DB.
* **Validation:** All chunks now have embeddings, and the L2 norms are \~1.0.

---

## Folder structure

```
runtime/
  data/raw/hotpot/        # HotpotQA slice
  staging/
    documents.jsonl
    chunks.jsonl
scripts/
  01_download_hotpot.py
  02_chunk_hotpot.py
  03_load_docs_chunks.py
  04_embed_chunks.py
```

---

## How I run it

### 0) Download the data

```bash
python scripts/01_download_hotpot.py
```

### 1) Chunk the data

```bash
python scripts/02_chunk_hotpot.py
```

### 2) Load into Postgres

```bash
python scripts/03_load_docs_chunks.py
```

### 3) Embed the chunks

```bash
python scripts/04_embed_chunks.py
```

At the end I check:

* All embeddings filled
* Norms around 1.0

---

## Next steps

1. Build an HNSW index for fast cosine search.
2. Make a small `/search` endpoint to try dense retrieval.
3. Add BM25 + RRF for hybrid search.
4. Add a reranker.
5. Build `/answer` that does multi‑hop retrieval with citations.
6. Add evaluation with RAGAS.

---

**Current status:** Chunking, loading, and embedding are done. Dense search is next.
