#!/usr/bin/env python3
"""
Simple CLI dense search over pgvector.

- Loads query, encodes with BAAI/bge-m3 (normalized)
- Searches `chunks` by cosine distance using the HNSW index (if created)
"""

import argparse
import textwrap
import sys
from typing import List, Tuple

import numpy as np
import psycopg2
from psycopg2 import sql
from sentence_transformers import SentenceTransformer
import torch


# ---------- Config ----------
DB = dict(
    host="localhost",
    port=5432,
    dbname="ragdb",
    user="rag",
    password="rag",
)

MODEL_NAME = "BAAI/bge-m3"


def vec_to_pgvector(v: np.ndarray) -> str:
    """Format numpy vector to pgvector text literal: '[v1,v2,...]'"""
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def load_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Model: {MODEL_NAME} | CUDA: {torch.cuda.is_available()} | Device: {model.device}")
    return model


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return emb


def search(conn, qvec_text: str, top_k: int) -> List[Tuple]:
    """Return rows: (id, doc_id, title, text, score) ordered by score desc."""
    with conn.cursor() as cur:
        # Use cosine distance (<=>). We convert to similarity for readability: 1 - distance
        query = sql.SQL(
            """
            SELECT id, doc_id, title, text,
                   1 - (embedding <=> {q}::vector) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> {q}::vector
            LIMIT {k}
            """
        ).format(q=sql.Literal(qvec_text), k=sql.Literal(top_k))

        cur.execute(query)
        return cur.fetchall()


def fmt_snippet(text: str, width: int = 100) -> str:
    snippet = text.replace("\n", " ").strip()
    if len(snippet) > width:
        snippet = snippet[: width - 1] + "…"
    return snippet


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Dense search over pgvector chunks",
    )
    ap.add_argument("query", help="Search query text")
    ap.add_argument("--k", "--top-k", dest="top_k", type=int, default=5, help="Number of results")
    args = ap.parse_args(argv)

    try:
        model = load_model()
    except Exception as e:
        print("Failed to load embedding model:", e, file=sys.stderr)
        return 2

    try:
        conn = psycopg2.connect(**DB)
    except Exception as e:
        print("Failed to connect to Postgres:", e, file=sys.stderr)
        print("Check host/port/db/user/password in scripts/search.py", file=sys.stderr)
        return 2

    with conn:
        qvec = embed_query(model, args.query)
        qvec_text = vec_to_pgvector(qvec)
        rows = search(conn, qvec_text, args.top_k)

    if not rows:
        print("No results.")
        return 0

    print(f"Top {args.top_k} results for: {args.query!r}\n")
    for i, (cid, doc_id, title, text, score) in enumerate(rows, 1):
        print(f"{i:>2}. score={score:.4f} | title={title}")
        print(f"    chunk_id={cid} doc_id={doc_id}")
        print(f"    {fmt_snippet(text, 160)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

