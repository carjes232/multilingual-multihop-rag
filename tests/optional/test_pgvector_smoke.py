import os

import numpy as np
import psycopg2
import pytest
from scripts.retriever import PgVectorRetriever

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_PGVECTOR_SMOKE"),
    reason="pgvector smoke is optional; set RUN_PGVECTOR_SMOKE=1 to enable.",
)


def _pg():
    return dict(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "ragdb"),
        user=os.getenv("DB_USER", "rag"),
        password=os.getenv("DB_PASSWORD", "rag"),
    )


def test_pgvector_roundtrip_smoke():
    cfg = _pg()
    with psycopg2.connect(**cfg) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            DROP TABLE IF EXISTS chunks;
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                title TEXT,
                text TEXT,
                embedding vector(1024)
            );
            """
        )
        # Insert two simple unit vectors
        v1 = np.zeros(1024, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(1024, dtype=np.float32)
        v2[1] = 1.0

        def vtext(v):
            return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"

        cur.execute(
            "INSERT INTO chunks (id, title, text, embedding) VALUES (%s,%s,%s,%s::vector)",
            ("c1", "T1", "First", vtext(v1)),
        )
        cur.execute(
            "INSERT INTO chunks (id, title, text, embedding) VALUES (%s,%s,%s,%s::vector)",
            ("c2", "T2", "Second", vtext(v2)),
        )
        conn.commit()

    r = PgVectorRetriever(cfg)
    qvec = np.expand_dims(v1, 0)
    results = r.search_vecs(qvec, k=1)
    assert results and len(results[0]) >= 1
    assert results[0][0].id == "c1"
