#!/usr/bin/env python3
"""
Retriever backends with environment gating: pgvector (Postgres) or Pinecone.

Env variables (examples shown also in .env.example):
  - RETRIEVER_BACKEND=pgvector | pinecone

Postgres (pgvector):
  - DB_HOST=localhost
  - DB_PORT=5432
  - DB_NAME=ragdb
  - DB_USER=rag
  - DB_PASSWORD=rag

Pinecone:
  - PINECONE_API_KEY=...
  - PINECONE_INDEX=rag-chunks
  - PINECONE_CLOUD=aws
  - PINECONE_REGION=us-east-1

Both backends expect BGE-M3 embeddings (1024-dim, normalized) for cosine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2 import sql

import embedder as emb
import dotenv

dotenv.load_dotenv()


try:  # pinecone client is optional
    # pinecone-client v3
    from pinecone import Pinecone
except Exception:  # pragma: no cover - optional at runtime if pgvector only
    Pinecone = None  # type: ignore
    



# -------- Data model --------
@dataclass
class Hit:
    id: str
    doc_id: Optional[str]
    title: Optional[str]
    text: str
    score: float


# -------- Env helpers --------
def get_backend() -> str:
    return (os.getenv("RETRIEVER_BACKEND") or "pgvector").strip().lower()


def get_db_cfg() -> dict:
    return dict(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "ragdb"),
        user=os.getenv("DB_USER", "rag"),
        password=os.getenv("DB_PASSWORD", "rag"),
    )


def vec_to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def embed_queries(texts: List[str]) -> np.ndarray:
    return emb.embed(texts, purpose="query")


# -------- pgvector backend --------
class PgVectorRetriever:
    def __init__(self, db_cfg: Optional[dict] = None):
        self.db_cfg = db_cfg or get_db_cfg()

    def _search_once(self, cur, qvec_text: str, k: int) -> List[Hit]:
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
        return [Hit(id=r[0], doc_id=r[1], title=r[2], text=r[3], score=float(r[4])) for r in rows]

    def search(self, model, queries: List[str], k: int) -> List[List[Hit]]:
        # 'model' parameter kept for backward compatibility; embedding is env-gated
        vecs = embed_queries(queries)
        return self.search_vecs(vecs, k)

    def search_vecs(self, vecs: np.ndarray, k: int) -> List[List[Hit]]:
        with psycopg2.connect(**self.db_cfg) as conn, conn.cursor() as cur:
            out: List[List[Hit]] = []
            for v in vecs:
                out.append(self._search_once(cur, vec_to_pgvector(v), k))
            return out


# -------- Pinecone backend --------
class PineconeRetriever:
    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        if Pinecone is None:
            raise RuntimeError("pinecone-client not installed; add it to requirements.txt")
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        self._pc = Pinecone(api_key=api_key)
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "rag-chunks")
        self._index = self._pc.Index(self.index_name)
        self._namespace = os.getenv("PINECONE_NAMESPACE", "")

    def search(self, model, queries: List[str], k: int) -> List[List[Hit]]:
        # Prefer integrated text search if available (no client-side embedding)
        out: List[List[Hit]] = []
        for q in queries:
            matches = None
            # Prefer integrated text search when available
            try:
                if hasattr(self._index, "search"):
                    res = self._index.search(
                        namespace=self._namespace or None,
                        query={"inputs": {"text": q}, "top_k": int(k)},
                    )
                    matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", None)
            except Exception:
                matches = None

            if matches is None:
                # Fallback: embed locally (same model name), then query by vector
                vecs = embed_queries([q])
                hits_batch = self.search_vecs(vecs, k)
                out.append(hits_batch[0] if hits_batch else [])
                continue

            ids = []
            scores = []
            for m in matches:
                mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", "")
                sc = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
                ids.append(str(mid))
                scores.append(float(sc) if sc is not None else 0.0)
            out.append(self._hydrate_hits(ids, scores))
        return out

    def search_vecs(self, vecs: np.ndarray, k: int) -> List[List[Hit]]:
        # Pinecone expects plain lists
        vec_lists = [v.tolist() for v in vecs]
        res = self._index.query(vector=vec_lists, top_k=k, include_metadata=True)
        # Batch query returns a list of QueryResponse
        out: List[List[Hit]] = []
        results = res.get("results") if isinstance(res, dict) else getattr(res, "results", None)
        if results is None:
            # Legacy single-query shape
            results = [res]
        for qr in results:
            matches = qr.get("matches") if isinstance(qr, dict) else getattr(qr, "matches", [])
            hits: List[Hit] = []
            for m in matches or []:
                mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", "")
                score = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
                md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
                # If metadata not present, hydrate from Postgres
                if not md:
                    hits.append(Hit(id=str(mid), doc_id=None, title=None, text="", score=float(score or 0.0)))
                else:
                    hits.append(Hit(
                        id=str(mid),
                        doc_id=str(md.get("doc_id")) if md and md.get("doc_id") is not None else None,
                        title=str(md.get("title")) if md and md.get("title") is not None else None,
                        text=str(md.get("text")) if md and md.get("text") is not None else "",
                        score=float(score) if score is not None else 0.0,
                    ))
            out.append(hits)
        # If any batch lacks text/title, hydrate from Postgres
        out_hydrated: List[List[Hit]] = []
        for hits in out:
            if any((not h.text) or (h.title is None) for h in hits):
                ids = [h.id for h in hits]
                scores = [h.score for h in hits]
                out_hydrated.append(self._hydrate_hits(ids, scores))
            else:
                out_hydrated.append(hits)
        return out_hydrated

    def _hydrate_hits(self, ids: List[str], scores: List[float]) -> List[Hit]:
        # Fetch chunk details from Postgres by id list
        try:
            cfg = get_db_cfg()
            with psycopg2.connect(**cfg) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id, doc_id, title, text FROM chunks WHERE id = ANY(%s)",
                    (ids,),
                )
                rows = cur.fetchall()
        except Exception:
            rows = []
        by_id = {str(r[0]): (r[1], r[2], r[3]) for r in rows}
        hydrated: List[Hit] = []
        for i, cid in enumerate(ids):
            doc_id, title, text = by_id.get(cid, (None, None, ""))
            hydrated.append(Hit(id=cid, doc_id=doc_id, title=title, text=text or "", score=scores[i]))
        return hydrated


# -------- Factory + pooling --------
def make_retriever() -> PgVectorRetriever | PineconeRetriever:
    backend = get_backend()
    if backend == "pinecone":
        return PineconeRetriever()
    return PgVectorRetriever()


def pool_and_rerank(batches: List[List[Hit]], top_k: int, max_pool: int = 24) -> List[Hit]:
    pooled: dict[str, Hit] = {}
    for hits in batches:
        for h in hits:
            prev = pooled.get(h.id)
            if (prev is None) or (h.score > prev.score):
                pooled[h.id] = h
    rows_sorted = sorted(pooled.values(), key=lambda x: x.score, reverse=True)
    return rows_sorted[: min(max_pool, top_k)]
