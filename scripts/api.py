#!/usr/bin/env python3
"""
FastAPI app exposing /search backed by Postgres + pgvector.

Run:
  uvicorn scripts.api:app --reload --port 8000

Env/Config: adjust DB dict below if needed.
"""

from typing import List

import numpy as np
import psycopg2
from psycopg2 import sql
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch


# -------- Config --------
DB = dict(
    host="localhost",
    port=5432,
    dbname="ragdb",
    user="rag",
    password="rag",
)
MODEL_NAME = "BAAI/bge-m3"


# -------- App state --------
app = FastAPI(title="RAG Search API")
_model: SentenceTransformer | None = None


def vec_to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def ensure_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"Model loaded: {MODEL_NAME} | CUDA: {torch.cuda.is_available()} | Device: {_model.device}")
    return _model


class SearchHit(BaseModel):
    id: str
    doc_id: str | None
    title: str | None
    text: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search", response_model=List[SearchHit])
def search(q: str = Query(..., description="Query text"), k: int = Query(5, ge=1, le=50)):
    model = ensure_model()
    qvec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    qvec_text = vec_to_pgvector(qvec)

    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        query = sql.SQL(
            """
            SELECT id, doc_id, title, text, 1 - (embedding <=> {q}::vector) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> {q}::vector
            LIMIT {k}
            """
        ).format(q=sql.Literal(qvec_text), k=sql.Literal(k))
        cur.execute(query)
        rows = cur.fetchall()

    hits = [
        SearchHit(id=r[0], doc_id=r[1], title=r[2], text=r[3], score=float(r[4]))
        for r in rows
    ]
    return hits

