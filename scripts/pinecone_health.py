#!/usr/bin/env python3
"""
Pinecone health + smoke search (integrated index friendly).

- Loads env from .env via python-dotenv.
- Checks index describe (best-effort) and prints embedding model info if present.
- Runs a few text queries using INTEGRATED SEARCH:
    index.search(namespace=..., query={"inputs": {"text": q}, "top_k": K}, fields=[...])
- If integrated search isn't available, falls back to local vector search using scripts/embedder.py.
- Normalizes responses from both old/new SDK shapes.

Env (read via dotenv, but can be overridden by CLI flags):
  PINECONE_API_KEY      (required)
  PINECONE_INDEX        (index name; ignored if PINECONE_INDEX_HOST is set)
  PINECONE_INDEX_HOST   (preferred for data ops; e.g., "your-index-xxxx.svc....pinecone.io")
  PINECONE_NAMESPACE    (optional; defaults to __default__)
  EMBEDDING_MODEL       (display only; e.g., multilingual-e5-large)

Usage examples:
  python scripts/pinecone_health.py
  PINECONE_INDEX=rag-integrated PINECONE_NAMESPACE=hotpot python scripts/pinecone_health.py --n 5 --k 5
  python scripts/pinecone_health.py --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl --n 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv

load_dotenv()


# ----------------------------
# Helpers
# ----------------------------
def load_queries_from_file(path: str, n: int) -> List[str]:
    qs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
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


def compact(s: str, max_chars: int = 160) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def try_describe_index(pc, name: Optional[str]) -> Dict:
    """Return a dict-ish description if possible; tolerate SDK/object forms."""
    if not name:
        return {"warn": "No index name (host-only mode). Skipping describe_index()."}
    try:
        desc = pc.describe_index(name=name)
        if isinstance(desc, dict):
            return desc
        # Object -> to dict (best-effort)
        out = {}
        for k in dir(desc):
            if k.startswith("_"):
                continue
            try:
                out[k] = getattr(desc, k)
            except Exception:
                pass
        # Some SDKs expose a to_dict()
        try:
            if hasattr(desc, "to_dict"):
                out.update(desc.to_dict())
        except Exception:
            pass
        return out
    except Exception as e:
        return {"error": str(e)}


def parse_hits(res) -> List[Tuple[str, float, Dict]]:
    """
    Normalize both new and old response shapes:
      - New: res.result.hits -> each hit has _id/_score/fields
      - Old: res.matches     -> each match has id/score/metadata
    Returns: [(id, score, fields_or_metadata_dict), ...]
    """
    # Dict-like
    if isinstance(res, dict):
        result = res.get("result")
        hits = None
        if isinstance(result, dict):
            hits = result.get("hits")
        if hits is None:
            hits = res.get("matches")
    else:
        # Object-like
        result = getattr(res, "result", None)
        hits = getattr(result, "hits", None) if result is not None else getattr(res, "matches", None)

    out: List[Tuple[str, float, Dict]] = []
    for h in hits or []:
        if isinstance(h, dict):
            _id = h.get("_id") or h.get("id") or ""
            _score = h.get("_score") or h.get("score") or 0.0
            f = h.get("fields") or h.get("metadata") or {}
        else:
            _id = getattr(h, "_id", getattr(h, "id", "")) or ""
            _score = getattr(h, "_score", getattr(h, "score", 0.0)) or 0.0
            f = getattr(h, "fields", getattr(h, "metadata", {})) or {}
        try:
            _score = float(_score)
        except Exception:
            _score = 0.0
        out.append((str(_id), _score, f if isinstance(f, dict) else {}))
    return out


def search_integrated(index, q: str, k: int, namespace: Optional[str], fields: Optional[Sequence[str]]):
    """
    Integrated text search. Returns normalized hits or None on API errors.
    """
    try:
        payload = {"inputs": {"text": q}, "top_k": int(k)}
        kwargs = {}
        if fields:
            kwargs["fields"] = list(fields)
        res = index.search(namespace=namespace or "__default__", query=payload, **kwargs)
        return parse_hits(res)
    except Exception:
        return None


def search_vector(index, q: str, k: int, namespace: Optional[str], fields: Optional[Sequence[str]]):
    """
    Fallback: embed locally (scripts/embedder.py) and search by vector.
    Returns normalized hits or None if embedder not available.
    """
    # Try a couple of import paths; run from repo root.
    emb = None
    try:
        import scripts.embedder as emb  # type: ignore
    except Exception:
        try:
            from . import embedder as emb  # type: ignore
        except Exception:
            return None

    try:
        vec = emb.embed([q], purpose="query")[0]
        payload = {"vector": {"values": vec.tolist()}, "top_k": int(k)}
        kwargs = {}
        if fields:
            kwargs["fields"] = list(fields)
        res = index.search(namespace=namespace or "__default__", query=payload, **kwargs)
        return parse_hits(res)
    except Exception:
        return None


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Pinecone health + smoke search (integrated index friendly)")
    ap.add_argument("--index", default=os.getenv("PINECONE_INDEX", "").strip(),
                    help="Index name (ignored if PINECONE_INDEX_HOST is set)")
    ap.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "__default__").strip())
    ap.add_argument("--file", default=None, help="JSONL with a 'question' or 'query' field to sample from")
    ap.add_argument("--n", type=int, default=5, help="Number of sample queries")
    ap.add_argument("--k", type=int, default=5, help="Top-K per query")
    ap.add_argument("--fields", default=os.getenv("PINECONE_FIELDS", "chunk_text,title,source,lang"),
                    help="Comma-separated list of fields to return (for printing)")
    args = ap.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        print("PINECONE_API_KEY is not set", file=sys.stderr)
        return 2

    index_host = os.getenv("PINECONE_INDEX_HOST", "").strip()
    expected_model = os.getenv("EMBEDDING_MODEL", "multilingual-e5-large").strip()

    try:
        from pinecone import Pinecone  # type: ignore
    except Exception as e:
        print("pinecone client not installed:", e, file=sys.stderr)
        return 2

    pc = Pinecone(api_key=api_key)

    # Open index by host (preferred) or by name
    index = pc.Index(host=index_host) if index_host else pc.Index(args.index)

    # Best-effort describe (only if we have a name)
    desc = try_describe_index(pc, None if index_host else args.index)
    src_model = None
    # Common keys across SDKs; may be absent
    for key in ("source_model", "sourceModel", "sourceModelName", "embed", "embedding", "model"):
        val = desc.get(key) if isinstance(desc, dict) else None
        if val:
            src_model = val
            break

    print("Index:",
          (args.index or "<by-host>"),
          "| host:",
          (index_host or "<by-name>"),
          "| namespace:",
          (args.namespace or "__default__"))
    if isinstance(desc, dict):
        print("Describe keys:", ", ".join(sorted(desc.keys())) or "<none>")
    print("Model on index (if reported):", src_model or "<not reported>", "| expected:", expected_model)

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

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    ok = 0
    for q in queries:
        print("\nQ:", q)
        matches = search_integrated(index, q, args.k, args.namespace, fields=fields)
        path = "integrated"
        if matches is None:  # API error (e.g., non-integrated index)
            matches = search_vector(index, q, args.k, args.namespace, fields=fields)
            path = "vector-fallback"

        if not matches:
            print("  No matches (path:", path + ")")
            continue

        # Print top up to min(3, k)
        for mid, sc, md in matches[: min(3, args.k)]:
            # For integrated search, text lives under 'fields'; for old API under 'metadata'
            title = (md or {}).get("title") or "<no title>"
            text = (md or {}).get("chunk_text") or (md or {}).get("text") or ""
            print(f"  - id={mid} score={sc:.4f} title={title}")
            if text:
                print("    ", compact(text))

        ok += 1

    print(f"\nSummary: {ok}/{len(queries)} queries returned matches.")
    if ok == 0:
        print("[warn] No matches. Check that:")
        print("  • You ran integrated ingestion (upsert_records) into THIS index/namespace.")
        print("  • The index is configured with an embedding model (or set PINECONE_INDEX_HOST).")
        print("  • If using vector-fallback, scripts/embedder.py is importable and configured.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
