#!/usr/bin/env python3
"""
FastAPI app exposing /search and /answer (RAG) with robust Ollama calls.

Run:
  uvicorn scripts.api:app --reload --port 8000
"""

from typing import List, Tuple, Optional
import re, json, time

import numpy as np
import psycopg2
from psycopg2 import sql
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import requests

# -------- Config --------
DB = dict(host="localhost", port=5432, dbname="ragdb", user="rag", password="rag")
MODEL_NAME = "BAAI/bge-m3"

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_GEN_URL  = "http://127.0.0.1:11434/api/generate"

# -------- App state --------
app = FastAPI(title="RAG Search API")
_model: Optional[SentenceTransformer] = None

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
def search(q: str = Query(...), k: int = Query(5, ge=1, le=50)):
    model = ensure_model()
    qvec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    qvec_text = vec_to_pgvector(qvec)

    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        query = sql.SQL("""
            SELECT id, doc_id, title, text, 1 - (embedding <=> {q}::vector) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> {q}::vector
            LIMIT {k}
        """).format(q=sql.Literal(qvec_text), k=sql.Literal(k))
        cur.execute(query)
        rows = cur.fetchall()

    hits = [SearchHit(id=r[0], doc_id=r[1], title=r[2], text=r[3], score=float(r[4])) for r in rows]
    return hits

# -------- Answer generation --------
class AnswerResponse(BaseModel):
    answer: str
    hits: List[SearchHit]
    latency_ms: int

def _compact(s: str, max_chars: int = 480) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:max_chars]

def build_prompt(question: str, hits: List[SearchHit]) -> str:
    lines = []
    lines.append("You are a precise QA assistant. Use ONLY the sources below.")
    lines.append("- If unknown, reply with the shortest possible 'unknown'.")
    lines.append("- Prefer exact minimal spans from sources.")
    lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Sources:")
    for i, h in enumerate(hits, 1):
        title = h.title or "Untitled"
        snippet = _compact(h.text)
        lines.append(f"[{i}] {title}: {snippet}")
    lines.append("")
    lines.append("Answer (short):")
    return "\n".join(lines)

def _strip_think(txt: str) -> str:
    return re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip()

def _shorten_final(txt: str) -> str:
    """Post-process to a short span: prefer yes/no; else first ≤6 words/line."""
    s = _strip_think(txt)
    m = re.match(r"\s*(yes|no)\b", s, flags=re.I)
    if m:
        return m.group(1).lower()
    # if JSON present
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "final" in obj:
            return str(obj["final"]).strip()
    except Exception:
        pass
    # otherwise keep it super short
    words = re.split(r"\s+", s)
    return " ".join(words[:6]).strip()

def _chat(payload: dict, timeout: int) -> Tuple[bool, dict, str]:
    try:
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
        if r.status_code == 200:
            return True, r.json(), ""
        return False, {}, f"{r.status_code} {r.text[:200]}"
    except requests.RequestException as e:
        return False, {}, str(e)

def _generate(payload: dict, timeout: int) -> Tuple[bool, dict, str]:
    try:
        r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=timeout)
        if r.status_code == 200:
            return True, r.json(), ""
        return False, {}, f"{r.status_code} {r.text[:200]}"
    except requests.RequestException as e:
        return False, {}, str(e)

def call_ollama(model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 64, timeout: int = 60) -> str:
    """
    Robust call that works across models (Qwen, Gemma, DeepSeek), even if JSON schema
    or 'think' isn't supported. Tries a cascade of fallbacks.
    Returns a short final string.
    """
    # 1) Chat + JSON Schema + think
    schema = {
        "type": "object",
        "properties": {"final": {"type": "string", "minLength": 1, "maxLength": 80}},
        "required": ["final"],
    }
    system = ("You may include a brief <think> plan (≤2 short sentences). "
              "Return ONLY a JSON object {\"final\": \"<short>\"}. If unsure, return \"unknown\".")
    trials = [
        ("chat_schema_think", {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "think": True,
            "format": schema,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _chat),
        # 2) Chat + JSON Schema (no think)
        ("chat_schema", {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _chat),
        # 3) Chat + "json" string format (no think)
        ("chat_json", {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _chat),
        # 4) Chat plain (no format/think)
        ("chat_plain", {
            "model": model,
            "messages": [{"role": "system", "content": "Answer concisely with only the final short answer."},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _chat),
        # 5) Generate + json
        ("gen_json", {
            "model": model,
            "prompt": prompt + "\n\nReturn ONLY JSON: {\"final\": \"<short>\"}",
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _generate),
        # 6) Generate plain
        ("gen_plain", {
            "model": model,
            "prompt": prompt + "\n\nReturn only the short final answer.",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _generate),
    ]

    last_err = ""
    for name, payload, func in trials:
        ok, data, err = func(payload, timeout)
        if not ok:
            last_err = f"{name}: {err}"
            continue
        # parse
        content = ""
        if "message" in data:
            content = (data.get("message") or {}).get("content", "") or data.get("response", "")
        else:
            content = data.get("response", "")
        content = _strip_think(content)
        # JSON path
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "final" in obj:
                return str(obj["final"]).strip()
        except Exception:
            pass
        # fallback: shorten
        return _shorten_final(content)

    raise HTTPException(status_code=502, detail=f"Ollama failed all fallbacks: {last_err}")

@app.get("/answer", response_model=AnswerResponse)
def answer(
    q: str = Query(..., description="Question"),
    k: int = Query(4, ge=1, le=10, description="Top-k contexts"),
    model: str = Query("qwen3:8b", description="Ollama model name"),
    temperature: float = Query(0.2, ge=0.0, le=1.0),
    max_tokens: int = Query(64, ge=16, le=512),
):
    # Retrieve
    search_hits = search(q, k)
    if not search_hits:
        raise HTTPException(status_code=404, detail="No contexts found")

    # Build prompt and call Ollama
    prompt = build_prompt(q, search_hits)
    t0 = time.time()
    ans = call_ollama(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    ms = int((time.time() - t0) * 1000)

    return AnswerResponse(answer=ans, hits=search_hits, latency_ms=ms)
