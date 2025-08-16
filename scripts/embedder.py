#!/usr/bin/env python3
"""
Embedding providers with environment gating.

Env:
  EMBEDDING_BACKEND=local | pinecone
  EMBEDDING_MODEL=
    - local: HuggingFace/SBERT model id (default: BAAI/bge-m3)
    - pinecone: Pinecone Inference model (default: multilingual-e5-large)

For E5-family models, we add the recommended prefixes:
  - Queries:  "query: <text>"
  - Passages: "passage: <text>"
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

_LOCAL_MODEL = None  # lazy-loaded SentenceTransformer
_LOCAL_MODEL_NAME = None
_PC = None           # lazy-initialized Pinecone client


def get_backend() -> str:
    return (os.getenv("EMBEDDING_BACKEND") or "local").strip().lower()


def get_model_name() -> str:
    b = get_backend()
    if b == "pinecone":
        return os.getenv("EMBEDDING_MODEL", "multilingual-e5-large")
    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def _maybe_prefix_e5(texts: List[str], purpose: str, model_name: str) -> List[str]:
    m = model_name.lower()
    if "e5" in m:
        pref = "query: " if purpose == "query" else "passage: "
        return [pref + (t or "") for t in texts]
    return texts


def _resolve_local_model_name(name: str) -> str:
    n = (name or "").strip()
    if n.lower() in {"multilingual-e5-large", "e5-multilingual-large", "intfloat/multilingual-e5-large"}:
        return "intfloat/multilingual-e5-large"
    return n


def _ensure_local_model():
    global _LOCAL_MODEL, _LOCAL_MODEL_NAME
    desired = _resolve_local_model_name(get_model_name())
    if _LOCAL_MODEL is None or _LOCAL_MODEL_NAME != desired:
        from sentence_transformers import SentenceTransformer  # lazy import
        import torch
        device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        _LOCAL_MODEL = SentenceTransformer(desired, device=device)
        _LOCAL_MODEL_NAME = desired
        print(f"Embedder(local): model={desired} device={getattr(_LOCAL_MODEL,'device','cpu')}")
    return _LOCAL_MODEL


def _ensure_pc():
    global _PC
    if _PC is None:
        from pinecone import Pinecone  # type: ignore
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set for Pinecone embedder")
        _PC = Pinecone(api_key=api_key)
        print(f"Embedder(pinecone): model={get_model_name()}")
    return _PC


def embed(texts: List[str], purpose: str = "query") -> np.ndarray:
    """Return L2-normalized embeddings (numpy array: [N, D])."""
    backend = get_backend()
    model_name = get_model_name()
    texts = _maybe_prefix_e5(texts, purpose, model_name)

    if backend == "pinecone":
        # Fallback to local SentenceTransformer with the same model name
        # so embeddings match Pinecone's hosted model output family.
        model = _ensure_local_model()
        arr = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype(np.float32)
    else:
        model = _ensure_local_model()
        arr = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # Ensure normalized output
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype(np.float32)
