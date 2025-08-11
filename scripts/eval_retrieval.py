#!/usr/bin/env python3
"""
Quick retrieval eval: recall@k using Hotpot titles as gold.

Assumes you have loaded/embedded chunks and created the HNSW index.

Usage:
  python scripts/eval_retrieval.py --k 5 --limit 50 \
    --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl
"""

import argparse
import json
from typing import List, Tuple, Set

import numpy as np
import psycopg2
from psycopg2 import sql
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


def vec_to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def load_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model


def db_search(conn, qvec_text: str, top_k: int) -> List[Tuple[str, str, str, str, float]]:
    with conn.cursor() as cur:
        query = sql.SQL(
            """
            SELECT id, doc_id, title, text, 1 - (embedding <=> {q}::vector) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> {q}::vector
            LIMIT {k}
            """
        ).format(q=sql.Literal(qvec_text), k=sql.Literal(top_k))
        cur.execute(query)
        return cur.fetchall()


def gold_titles_from_example(ex: dict) -> Set[str]:
    ctx = ex.get("context", {})
    titles = ctx.get("title", [])
    if isinstance(titles, list):
        return {str(t).strip() for t in titles if t}
    return set()


def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval recall@k using HotpotQA titles")
    ap.add_argument("--file", default="runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl",
                    help="Hotpot JSONL with fields question/answer/context")
    ap.add_argument("--k", dest="top_k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=50, help="How many questions to evaluate")
    args = ap.parse_args()

    model = load_model()
    conn = psycopg2.connect(**DB)

    total = 0
    hits = 0
    examples = []

    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            if args.limit and total >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            q = ex.get("question")
            gold_titles = gold_titles_from_example(ex)
            if not q or not gold_titles:
                continue

            qvec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
            rows = db_search(conn, vec_to_pgvector(qvec), args.top_k)
            ret_titles = {r[2] for r in rows if r[2]}
            is_hit = len(gold_titles & ret_titles) > 0
            hits += 1 if is_hit else 0
            total += 1
            if total <= 5:  # keep a few samples for display
                examples.append((q, list(gold_titles)[:3], [(r[2], float(r[4])) for r in rows[:3]], is_hit))

    recall = hits / total if total else 0.0

    print(f"Evaluated: {total} questions | k={args.top_k}")
    print(f"Recall@{args.top_k}: {recall:.3f}  (hits={hits})\n")
    print("Examples:")
    for i, (q, gold, retrieved, ok) in enumerate(examples, 1):
        print(f"{i}. {'HIT ' if ok else 'MISS'} q={q}")
        print(f"   gold_titles: {gold}")
        print(f"   top3: {retrieved}\n")

    conn.close()


if __name__ == "__main__":
    main()

