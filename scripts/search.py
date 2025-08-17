#!/usr/bin/env python3
"""
Simple CLI dense search with env-gated backend (pgvector or Pinecone).

- Loads query, encodes with BAAI/bge-m3 (normalized)
- Uses Postgres pgvector by default; set RETRIEVER_BACKEND=pinecone to use Pinecone
"""

import argparse
import os
import sys

import embedder as emb
import retriever as retr

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def maybe_load_model():
    if emb.get_backend() == "local":
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"Model: {MODEL_NAME} | CUDA: {torch.cuda.is_available()} | Device: {model.device}")
        return model
    else:
        print(f"Using Pinecone embedding model: {emb.get_model_name()}")
        return None


def fmt_snippet(text: str, width: int = 100) -> str:
    snippet = text.replace("\n", " ").strip()
    if len(snippet) > width:
        snippet = snippet[: width - 1] + "…"
    return snippet


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dense search over chunks (pgvector or Pinecone)")
    ap.add_argument("query", help="Search query text")
    ap.add_argument("--k", "--top-k", dest="top_k", type=int, default=5, help="Number of results")
    args = ap.parse_args(argv)

    model = None
    if emb.get_backend() == "local":
        try:
            model = maybe_load_model()
        except Exception as e:
            print("Failed to load embedding model:", e, file=sys.stderr)
            return 2

    backend = retr.get_backend()
    try:
        r = retr.make_retriever()
    except Exception as e:
        print(f"Failed to init retriever backend '{backend}':", e, file=sys.stderr)
        return 2

    batches = r.search(model, [args.query], args.top_k)
    hits = batches[0] if batches else []

    if not hits:
        print("No results.")
        return 0

    print(f"Top {args.top_k} results for: {args.query!r}\n")
    for i, h in enumerate(hits, 1):
        print(f"{i:>2}. score={h.score:.4f} | title={h.title}")
        print(f"    chunk_id={h.id} doc_id={h.doc_id}")
        print(f"    {fmt_snippet(h.text, 160)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
