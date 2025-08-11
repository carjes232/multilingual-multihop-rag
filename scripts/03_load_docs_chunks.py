#!/usr/bin/env python3
import json
import pathlib
import psycopg2
from psycopg2.extras import execute_values, Json

# ---------- Config ----------
ROOT = pathlib.Path(__file__).parent.resolve().parent
PATH_DOCS = ROOT / "runtime" / "staging" / "documents.jsonl"
PATH_CHUNKS = ROOT / "runtime" / "staging" / "chunks.jsonl"

DB = dict(
    host="localhost",   # if your Python runs on Windows host and Postgres is port-mapped
    port=5432,
    dbname="ragdb",
    user="rag",
    password="rag",
)

BATCH = 1000  # tune if you like

# ---------- Connect ----------
conn = psycopg2.connect(**DB)
cur = conn.cursor()

# ---------- Ensure extension ----------
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# ---------- Recreate schema ----------
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id    TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    lang  TEXT,
    url   TEXT,
    meta  JSONB
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id        TEXT PRIMARY KEY,
    doc_id    TEXT REFERENCES documents(id),
    ord       INTEGER,
    lang      TEXT,
    title     TEXT,
    text      TEXT,
    tokens    INTEGER,
    embedding VECTOR(1024)  -- stays NULL until we embed
);
""")
conn.commit()

# ---------- Load documents ----------
insert_docs_sql = """
INSERT INTO documents (id, title, lang, url, meta)
VALUES %s
ON CONFLICT (id) DO UPDATE
SET title = EXCLUDED.title,
    lang  = EXCLUDED.lang,
    url   = EXCLUDED.url,
    meta  = EXCLUDED.meta
"""

rows = []
docs_lines = 0
with open(PATH_DOCS, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append((
            d["id"],
            d.get("title"),
            d.get("lang"),
            d.get("url"),
            Json(d.get("meta") or {}),
        ))
        docs_lines += 1
        if len(rows) >= BATCH:
            execute_values(cur, insert_docs_sql, rows, page_size=BATCH)
            conn.commit()
            rows.clear()

if rows:
    execute_values(cur, insert_docs_sql, rows, page_size=BATCH)
    conn.commit()
    rows.clear()

# ---------- Load chunks ----------
insert_chunks_sql = """
INSERT INTO chunks (id, doc_id, ord, lang, title, text, tokens, embedding)
VALUES %s
ON CONFLICT (id) DO UPDATE
SET doc_id = EXCLUDED.doc_id,
    ord    = EXCLUDED.ord,
    lang   = EXCLUDED.lang,
    title  = EXCLUDED.title,
    text   = EXCLUDED.text,
    tokens = EXCLUDED.tokens
    -- embedding stays as-is unless you explicitly update it
"""

rows = []
chunks_lines = 0
with open(PATH_CHUNKS, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        rows.append((
            c["id"],
            c.get("doc_id"),
            c.get("ord"),
            c.get("lang"),
            c.get("title"),
            c.get("text"),
            c.get("tokens"),
            None,  # embedding stays NULL for now
        ))
        chunks_lines += 1
        if len(rows) >= BATCH:
            execute_values(cur, insert_chunks_sql, rows, page_size=BATCH)
            conn.commit()
            rows.clear()

if rows:
    execute_values(cur, insert_chunks_sql, rows, page_size=BATCH)
    conn.commit()
    rows.clear()

# ---------- Sanity prints ----------
cur.execute("SELECT COUNT(*) FROM documents;")
doc_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM chunks;")
chunk_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;")
emb_filled = cur.fetchone()[0]

print(f"documents.jsonl lines (seen): {docs_lines}")
print(f"chunks.jsonl lines (seen):    {chunks_lines}")
print(f"DB documents count:           {doc_count}")
print(f"DB chunks count:              {chunk_count}")
print(f"DB chunks with embeddings:    {emb_filled}")

cur.close()
conn.close()
