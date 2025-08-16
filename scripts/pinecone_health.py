#!/usr/bin/env python3
"""
Pinecone health + smoke search.

Checks that the index is configured for online embeddings (source_model) and
runs a few text queries via integrated `index.search(...)` to verify non-zero
results. Falls back to vector `query` only to surface a warning if records-only
data is present.

Usage examples:
  # Use default env vars and built-in queries
  python scripts/pinecone_health.py

  # Sample 5 queries from a Hotpot slice
  python scripts/pinecone_health.py --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl --n 5

Env:
  PINECONE_API_KEY (required)
  PINECONE_INDEX (default: rag-chunks)
  PINECONE_NAMESPACE (optional)
  EMBEDDING_MODEL (used to display expected source_model)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Tuple

import dotenv

dotenv.load_dotenv()


def load_queries_from_file(path: str, n: int) -> List[str]:
    qs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                q = obj.get("question") or obj.get("query")
                if isinstance(q, str) and q.strip():
                    qs.append(q.strip())
            except Exception:
                continue
    random.shuffle(qs)
    return qs[:n]


def compact(s: str, max_chars: int = 140) -> str:
    s = " ".join((s or "").split())
    return s[:max_chars]


def try_describe_index(pc, index_name: str) -> dict:
    try:
        desc = pc.describe_index(index_name)
        if isinstance(desc, dict):
            return desc
        # v3 SDK returns an object; convert to dict-like
        out = {k: getattr(desc, k) for k in dir(desc) if not k.startswith("_")}
        return out
    except Exception as e:
        return {"error": str(e)}


def search_integrated(index, q: str, k: int, namespace: str | None):
    # Returns list of (id, score, metadata) or None on failure
    try:
        if not hasattr(index, "search"):
            return None
        res = index.search(
            namespace=namespace,
            query={
                "inputs": {"text": q},
                "top_k": int(k),
                "include_metadata": True,
            },
        )
        matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", None)
        if matches is None:
            return None
        out = []
        for m in matches:
            if isinstance(m, dict):
                mid = m.get("id")
                sc = m.get("score")
                md = m.get("metadata") or {}
            else:
                mid = getattr(m, "id", "")
                sc = getattr(m, "score", 0.0)
                md = getattr(m, "metadata", {}) or {}
            out.append((str(mid), float(sc) if sc is not None else 0.0, md if isinstance(md, dict) else {}))
        return out
    except Exception:
        return None


def search_vector(index, q: str, k: int, namespace: str | None):
    # Fallback only: embeds locally through SentenceTransformers if available
    try:
        from . import embedder as emb  # type: ignore
    except Exception:
        try:
            import embedder as emb  # type: ignore
        except Exception:
            return None
    try:
        vec = emb.embed([q], purpose="query")[0]
        res = index.query(vector=vec.tolist(), top_k=int(k), include_metadata=True, namespace=namespace)
        matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])
        out = []
        for m in matches or []:
            if isinstance(m, dict):
                mid = m.get("id")
                sc = m.get("score")
                md = m.get("metadata") or {}
            else:
                mid = getattr(m, "id", "")
                sc = getattr(m, "score", 0.0)
                md = getattr(m, "metadata", {}) or {}
            out.append((str(mid), float(sc) if sc is not None else 0.0, md if isinstance(md, dict) else {}))
        return out
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Pinecone health + smoke search")
    ap.add_argument("--index", default=os.getenv("PINECONE_INDEX", "rag-chunks"))
    ap.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", None))
    ap.add_argument("--file", default=None, help="Optional JSONL with a 'question' field to sample from")
    ap.add_argument("--n", type=int, default=5, help="Number of sample queries to run")
    ap.add_argument("--k", type=int, default=5, help="Top-K results to print")
    args = ap.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY is not set", file=sys.stderr)
        return 2

    try:
        from pinecone import Pinecone  # type: ignore
    except Exception as e:
        print("pinecone client not installed:", e, file=sys.stderr)
        return 2

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index)

    # Describe index and look for source_model
    desc = try_describe_index(pc, args.index)
    src_model = None
    for key in ("source_model", "sourceModel", "sourceModelName"):
        val = desc.get(key) if isinstance(desc, dict) else None
        if val:
            src_model = val
            break
    expected = os.getenv("EMBEDDING_MODEL", "multilingual-e5-large")
    if expected.lower() in {"intfloat/multilingual-e5-large", "e5-multilingual-large"}:
        expected = "multilingual-e5-large"
    print("Index:", args.index, "namespace:", args.namespace or "<default>")
    print("Describe keys:", ", ".join(sorted(list(desc.keys()))) if isinstance(desc, dict) else type(desc))
    print("source_model:", src_model or "<not reported>", "| expected:", expected)

    # Prepare queries
    if args.file:
        queries = load_queries_from_file(args.file, args.n)
        if not queries:
            print("No queries found in file; using defaults.")
    else:
        queries = []
    if not queries:
        queries = [
            "Where is the Random House Tower located?",
            "Who wrote the novel '1984'?",
            "What is the capital of Portugal?",
            "When was the Apollo 11 mission?",
            "What is the speed of light?",
        ][: args.n]

    ok = 0
    for q in queries:
        print("\nQ:", q)
        matches = search_integrated(index, q, args.k, args.namespace)
        path = "integrated"
        if matches is None:
            matches = search_vector(index, q, args.k, args.namespace)
            path = "vector-fallback"

        if not matches:
            print("  No matches (path:", path + ")")
            continue
        # Print top 3
        c = 0
        for mid, sc, md in matches:
            title = (md or {}).get("title") or "<no title>"
            text = (md or {}).get("text") or ""
            print(f"  - id={mid} score={sc:.4f} title={title}")
            if text:
                print("    ", compact(text))
            c += 1
            if c >= min(3, args.k):
                break
        if matches:
            ok += 1

    print(f"\nSummary: {ok}/{len(queries)} queries returned matches.")
    if ok == 0:
        print("[warn] No matches found. Ensure index has source_model and records are upserted.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

