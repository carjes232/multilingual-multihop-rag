#!/usr/bin/env python3
"""
Upsert chunk embeddings into a Pinecone index.

Reads chunks from Postgres (id, doc_id, title, text), embeds with BGE-M3,
and upserts to Pinecone with metadata. Uses env for config:

  RETRIEVER_BACKEND=pinecone            # optional; this script always targets Pinecone
  PINECONE_API_KEY=...
  PINECONE_INDEX=rag-chunks
  PINECONE_CLOUD=aws                    # for index creation
  PINECONE_REGION=us-east-1             # for index creation

DB config (source of chunks):
  DB_HOST=localhost
  DB_PORT=5432
  DB_NAME=ragdb
  DB_USER=rag
  DB_PASSWORD=rag

Example:
  python scripts/04b_upsert_chunks_pinecone.py --limit 2000 --batch 128
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from pinecone import Pinecone, ServerlessSpec
import dotenv
import embedder as emb
import numpy as np

dotenv.load_dotenv()


EMBED_DIM_DEFAULT = 1024  


def get_db_cfg() -> dict:
    return dict(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "ragdb"),
        user=os.getenv("DB_USER", "rag"),
        password=os.getenv("DB_PASSWORD", "rag"),
    )


def ensure_pinecone_index(pc: Pinecone, name: str, source_model: str | None, dimension: int | None) -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}  # type: ignore[index]
    if name in existing:
        return
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    kwargs = dict(name=name, spec=ServerlessSpec(cloud=cloud, region=region))
    if source_model:
        kwargs["source_model"] = source_model
    else:
        kwargs["dimension"] = dimension or EMBED_DIM_DEFAULT
        kwargs["metric"] = "cosine"
    pc.create_index(**kwargs)  # type: ignore[arg-type]


def embed_batch(texts: List[str]) -> np.ndarray:
    # Use env-gated embedder; passages for corpus
    return emb.embed(texts, purpose="passage")


def main():
    ap = argparse.ArgumentParser(description="Upsert chunk embeddings to Pinecone from Postgres")
    default_index = os.getenv("PINECONE_INDEX", "rag-chunks-e5-multilingual")
    ap.add_argument("--index", default=default_index, help="Pinecone index name")
    ap.add_argument("--batch", type=int, default=128, help="DB fetch page size; upsert is sub-batched automatically")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of chunks (0 = all)")
    ap.add_argument("--create-index", action="store_true", help="Create Pinecone index if missing")
    args = ap.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)
    if args.create_index:
        try:
            ensure_pinecone_index(pc, args.index, source_model=os.getenv("EMBEDDING_MODEL", "multilingual-e5-large"), dimension=None)
        except TypeError:
            # Older SDK: fall back to classical index creation
            ensure_pinecone_index(pc, args.index, source_model=None, dimension=EMBED_DIM_DEFAULT)
    index = pc.Index(args.index)

    # Pinecone service imposes a maximum records-per-upsert; enforce a safe ceiling.
    # Default to 96 unless overridden by env var.
    try:
        max_upsert = int(os.getenv("PINECONE_MAX_BATCH", "96"))
        if max_upsert <= 0:
            max_upsert = 96
    except Exception:
        max_upsert = 96

    db = get_db_cfg()
    conn = psycopg2.connect(**db)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    seen = 0
    t0 = time.time()

    while True:
        # fetch a page of chunks
        cur.execute(
            """
            SELECT id, doc_id, title, text
            FROM chunks
            ORDER BY id
            LIMIT %s OFFSET %s
            """,
            (args.batch, seen),
        )
        rows = cur.fetchall()
        if not rows:
            break

        ids = [str(r["id"]) for r in rows]
        texts = [str(r["text"]) for r in rows]
        titles = [str(r["title"]) if r["title"] is not None else None for r in rows]
        doc_ids = [str(r["doc_id"]) if r["doc_id"] is not None else None for r in rows]

        # Prefer integrated upsert when available
        if hasattr(index, "upsert_records"):
            # Pinecone Inference integrated path: send text and flatten metadata fields at top-level.
            # Some deployments reject nested objects at key 'metadata'; keep only primitives.
            records = []
            for i in range(len(ids)):
                rec = {"id": ids[i], "text": texts[i]}
                if titles[i] is not None:
                    rec["title"] = titles[i]
                if doc_ids[i] is not None:
                    rec["doc_id"] = doc_ids[i]
                records.append(rec)

            ns = os.getenv("PINECONE_NAMESPACE", None)
            for j in range(0, len(records), max_upsert):
                chunk = records[j : j + max_upsert]
                index.upsert_records(namespace=ns, records=chunk)
        else:
            # Fallback: embed locally (same model name) and upsert vectors
            from embedder import embed as embed_fn  # local ST fallback
            vecs = embed_fn(texts, purpose="passage")
            vectors = []
            for i in range(len(ids)):
                md = {"text": texts[i]}
                if titles[i] is not None:
                    md["title"] = titles[i]
                if doc_ids[i] is not None:
                    md["doc_id"] = doc_ids[i]
                vectors.append({"id": ids[i], "values": vecs[i].tolist(), "metadata": md})

            ns = os.getenv("PINECONE_NAMESPACE", None)
            for j in range(0, len(vectors), max_upsert):
                chunk = vectors[j : j + max_upsert]
                index.upsert(vectors=chunk, namespace=ns)

        seen += len(rows)
        if args.limit and seen >= args.limit:
            break
        print(f"Upserted {seen} vectors...")

    dt = time.time() - t0
    print(f"Done. Upserted {seen} vectors to Pinecone index '{args.index}' in {dt:.1f}s")

    cur.close(); conn.close()


if __name__ == "__main__":
    main()
