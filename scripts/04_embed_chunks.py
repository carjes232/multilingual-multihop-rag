#!/usr/bin/env python3
"""
04_embed_chunks.py
- Fetch chunks with NULL embeddings from Postgres
- Embed with BAAI/bge-m3 on GPU (if available), L2-normalized
- Bulk-update embeddings into pgvector column
- Sanity-check norms at the end

Reads DB config from environment variables (see .env.example):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import math
import os
import time

import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
# Load environment (.env) if present
load_dotenv()
DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "ragdb"),
    user=os.getenv("DB_USER", "rag"),
    password=os.getenv("DB_PASSWORD", "rag"),
)
MODEL_NAME = "BAAI/bge-m3"
GPU_BATCH = 64  # texts per forward pass on GPU; lower if VRAM is tight
QUERY_LIMIT = 512  # rows pulled from DB per outer loop
PAGE_SIZE = 1000  # rows per VALUES page in bulk UPDATE


# ---------- Helpers ----------
def vec_to_pgvector(v: np.ndarray) -> str:
    """Format numpy vector to pgvector text literal: '[v1,v2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


# ---------- Load model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
print(f"CUDA available: {torch.cuda.is_available()} | Model device: {model.device}")

# ---------- DB connect ----------
conn = psycopg2.connect(**DB)
conn.autocommit = False
cur = conn.cursor()

# Ensure extension (idempotent)
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

total_updated = 0
t0 = time.time()

while True:
    # Pull a window of pending rows; lock them so parallel workers don't collide
    cur.execute(f"""
        SELECT id, text
        FROM chunks
        WHERE embedding IS NULL
        LIMIT {QUERY_LIMIT}
        FOR UPDATE SKIP LOCKED
    """)
    rows = cur.fetchall()
    if not rows:
        break

    ids, texts = zip(*rows)  # keeps order
    # Encode in GPU-sized minibatches
    batch_vecs = []
    for i in range(0, len(texts), GPU_BATCH):
        batch_texts = list(texts[i : i + GPU_BATCH])
        embs = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        batch_vecs.append(embs)
    embs_np = np.vstack(batch_vecs)  # shape: (N, 1024)

    # Prepare pairs (id, vector_text)
    values = [(ids[i], vec_to_pgvector(embs_np[i])) for i in range(len(ids))]

    # Bulk UPDATE using VALUES; cast to ::vector so pgvector type is used
    sql = """
        UPDATE chunks AS c
        SET embedding = v.emb
        FROM (VALUES %s) AS v(id, emb)
        WHERE c.id = v.id
    """
    execute_values(cur, sql, values, template="(%s, %s::vector)", page_size=PAGE_SIZE)
    conn.commit()

    total_updated += len(values)
    print(f"Updated {total_updated} embeddings...")

# ---------- Final sanity ----------
cur.execute("SELECT COUNT(*) FROM chunks;")
total_chunks = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL;")
missing = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;")
filled = cur.fetchone()[0]

print("\nSanity:")
print(f"- Total chunks:               {total_chunks}")
print(f"- Chunks with embeddings:     {filled}")
print(f"- Chunks still missing:       {missing}")


# Fetch a few embeddings and compute L2 norms in Python
def parse_pgvector_text(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [float(x) for x in s.split(",") if x]


cur.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 3;")
rows = cur.fetchall()
for _id, emb_raw in rows:
    emb = parse_pgvector_text(emb_raw) if isinstance(emb_raw, str) else emb_raw
    norm = math.sqrt(sum(x * x for x in emb))
    print(f"id={_id}  norm={norm:.6f}")
