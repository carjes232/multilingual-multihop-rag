#!/usr/bin/env python3
"""
FastAPI app exposing /search and /answer (RAG) with robust Ollama calls
and entity-aware multi-query retrieval.

Run:
  uvicorn scripts.api:app --reload --port 8000
"""

from typing import List, Tuple, Optional, Iterable
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
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"

# -------- App state --------
app = FastAPI(title="RAG Search API")
_model: Optional[SentenceTransformer] = None

# -------- Utilities --------
def vec_to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"

def ensure_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"Model loaded: {MODEL_NAME} | CUDA: {torch.cuda.is_available()} | Device: {_model.device}")
    return _model

def is_yesno_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return q.startswith((
        "is ","are ","was ","were ","do ","does ","did ",
        "can ","could ","should ","has ","have ","had ",
        "will ","would "
    ))

# --- entity extraction for multi-query retrieval ---
_QUOTED = re.compile(r'["“”](.+?)["“”]')
# Title Case / proper noun-ish spans (handles hyphens/apostrophes, e.g., Esma Sultan Mansion; Random House Tower)
_TITLE_SPAN = re.compile(r"\b([A-Z][A-Za-z'’\-0-9]*(?:\s+[A-Z][A-Za-z'’\-0-9]*)+)\b")
# Number + street/building like "888 7th Avenue"
_NUM_TITLE = re.compile(r"\b(\d{1,4}(?:st|nd|rd|th)?\s+[A-Z][A-Za-z0-9'’.\-]+(?:\s+[A-Z][A-Za-z0-9'’.\-]+)*)\b")

def extract_entities(q: str) -> List[str]:
    s = q or ""
    ents: set[str] = set()
    # quoted phrases first
    for m in _QUOTED.finditer(s):
        phrase = m.group(1).strip()
        if phrase: ents.add(phrase)
    # number+title patterns
    for m in _NUM_TITLE.finditer(s):
        t = m.group(1).strip()
        if t: ents.add(t)
    # title-case spans (filter out common non-entities)
    for m in _TITLE_SPAN.finditer(s):
        span = m.group(1).strip()
        if len(span.split()) >= 2:
            ents.add(span)
    # de-dup near duplicates by lowercase
    out = []
    seen = set()
    for e in ents:
        k = e.lower()
        if k not in seen:
            seen.add(k); out.append(e)
    return out

# -------- Data models --------
class SearchHit(BaseModel):
    id: str
    doc_id: str | None
    title: str | None
    text: str
    score: float

class AnswerResponse(BaseModel):
    answer: str
    hits: List[SearchHit]
    latency_ms: int

@app.get("/health")
def health():
    return {"status": "ok"}

# -------- Low-level PG search --------
def _pg_search(cur, qvec_text: str, k: int) -> List[tuple]:
    query = sql.SQL("""
        SELECT id, doc_id, title, text, 1 - (embedding <=> {q}::vector) AS score
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> {q}::vector
        LIMIT {k}
    """).format(q=sql.Literal(qvec_text), k=sql.Literal(k))
    cur.execute(query)
    return cur.fetchall()

# -------- Public /search (single-query, unchanged) --------
@app.get("/search", response_model=List[SearchHit])
def search(q: str = Query(...), k: int = Query(5, ge=1, le=50)):
    model = ensure_model()
    qvec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    qvec_text = vec_to_pgvector(qvec)
    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        rows = _pg_search(cur, qvec_text, k)
    return [SearchHit(id=r[0], doc_id=r[1], title=r[2], text=r[3], score=float(r[4])) for r in rows]

# -------- Internal: multi-query retrieval --------
def search_multi(question: str, k: int, k_per_entity: int = 3, max_pool: int = 24) -> List[SearchHit]:
    """
    Retrieve by:
      1) the full question (k),
      2) each detected entity (k_per_entity),
    merge by id keeping the best score, re-rank, then return ONLY top-k.
    """
    model = ensure_model()
    queries: List[tuple[str,int]] = [(question, max(2, k))]  # ensure at least 2 from the full question
    # entity queries (cap to avoid over-encoding)
    ents = extract_entities(question)[:6]
    for e in ents:
        queries.append((e, k_per_entity))

    # encode all queries
    texts = [q for q,_ in queries]
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    vec_texts = [vec_to_pgvector(v) for v in vecs]

    pooled: dict[str, tuple] = {}
    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        for (qtxt, _k), vtxt in zip(queries, vec_texts):
            rows = _pg_search(cur, vtxt, _k)
            for r in rows:
                rid = r[0]
                prev = pooled.get(rid)
                if (not prev) or (float(r[4]) > float(prev[4])):  # keep highest score
                    pooled[rid] = r

    # sort by score desc and clip to top-k (but don't exceed a modest pool)
    rows_sorted = sorted(pooled.values(), key=lambda r: float(r[4]), reverse=True)[:max_pool]
    final_rows = rows_sorted[:k]
    return [SearchHit(id=r[0], doc_id=r[1], title=r[2], text=r[3], score=float(r[4])) for r in final_rows]

# -------- Think stripping & parsing --------
_THINK_ANY = re.compile(r"(?is)<\s*(think|thought|thinking)\b[^>]*>.*?<\s*/\s*\1\s*>")
_THINK_OPEN = re.compile(r"(?is)<\s*(think|thought|thinking)\b[^>]*>.*\Z")
_CODEFENCE = re.compile(r"(?is)```(?:json|txt|markdown)?\s*(.*?)\s*```")

def _strip_think(txt: str) -> str:
    if not txt:
        return ""
    s = txt
    s = _THINK_ANY.sub("", s)
    s = _THINK_OPEN.sub("", s)
    mcf = _CODEFENCE.search(s)
    if mcf:
        s = mcf.group(1)
    return s.strip()

def _extract_final_short(s: str, max_words: int = 8) -> str:
    if not s:
        return ""
    t = _strip_think(s)

    # yes/no first word
    m = re.match(r"\s*(yes|no)\b", t, flags=re.I)
    if m: return m.group(1).lower()

    # strict JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "final" in obj:
            ans = str(obj["final"]).strip()
            if ans: return ans
    except Exception:
        pass

    # near-JSON "final: ..."
    m2 = re.search(r'(?is)\bfinal\b\s*:\s*["\']?([^\n\r"\'}]*)', t)
    if m2:
        cand = m2.group(1).strip()
        if cand: return cand

    # markers
    for pat in (r'(?is)\bfinal answer\b\s*:\s*(.+)',
                r'(?is)\banswer\s*\(short\)\s*:\s*(.+)',
                r'(?is)\banswer\b\s*:\s*(.+)'):
        mm = re.search(pat, t)
        if mm:
            words = re.split(r'\s+', re.sub(r'\s+', ' ', mm.group(1)).strip())
            return " ".join(words[:max_words]).strip()

    # fallback: last non-empty line, else first words
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        words = re.split(r"\s+", lines[-1])
        out = " ".join(words[:max_words]).strip()
        if out and not out.lower().startswith("<think"):
            return out
    return " ".join(re.split(r"\s+", t)[:max_words]).strip()

# -------- Ollama helpers --------
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

def list_ollama_models(timeout: int = 5) -> list[str]:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        models = data.get("models") or []
        names: list[str] = []
        for m in models:
            name = (m or {}).get("name")
            if isinstance(name, str):
                names.append(name)
        return names
    except requests.RequestException:
        return []

def _parse_ollama_response(data: dict) -> str:
    if "message" in data:
        return (data.get("message") or {}).get("content","") or data.get("response","") or ""
    return data.get("response","") or ""

def call_ollama(model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 80, timeout: int = 60) -> str:
    """
    Robust call that works across models (Qwen, Gemma, DeepSeek), including reasoning models.
    We try multiple strategies and only return a short, think-free final span.
    """
    schema = {
        "type": "object",
        "properties": {"final": {"type": "string", "minLength": 1, "maxLength": 80}},
        "required": ["final"],
    }
    sys_plain = "Answer concisely with only the final short answer. Do NOT include <think>."
    sys_json  = ("Return ONLY a JSON object {\"final\": \"<short>\"}. "
                 "Do NOT include <think> or explanations. If unsure, return \"unknown\".")

    trials = [
        ("chat_schema", {
            "model": model,
            "messages": [{"role": "system", "content": sys_json},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _chat),
        ("chat_json", {
            "model": model,
            "messages": [{"role": "system", "content": sys_json},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _chat),
        ("chat_plain", {
            "model": model,
            "messages": [{"role": "system", "content": sys_plain},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _chat),
        ("gen_json", {
            "model": model,
            "prompt": prompt + "\n\nReturn ONLY JSON: {\"final\": \"<short>\"}",
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _generate),
        ("gen_plain", {
            "model": model,
            "prompt": prompt + "\n\nReturn only the short final answer. Do NOT include <think>.",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _generate),
    ]

    errs: list[str] = []
    for name, payload, func in trials:
        ok, data, err = func(payload, timeout)
        if not ok:
            errs.append(f"{name}: {err}")
            continue

        raw = _parse_ollama_response(data)
        # try strict JSON first
        try:
            obj = json.loads(_strip_think(raw))
            if isinstance(obj, dict) and "final" in obj and str(obj["final"]).strip():
                return str(obj["final"]).strip()
        except Exception:
            pass

        # heuristic extractor
        ans = _extract_final_short(raw)
        if ans and not ans.lower().startswith("<think"):
            return ans

    detail = {"error": "Ollama failed to produce a usable short answer", "attempts": errs[:6]}
    raise HTTPException(status_code=502, detail=detail)

# -------- Prompt building --------
def _compact(s: str, max_chars: int = 480) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:max_chars]

def build_prompt(question: str, hits: List[SearchHit]) -> str:
    yn = is_yesno_question(question)
    lines = []
    lines.append("You are a precise QA assistant. Use ONLY the sources below.")
    lines.append("- Copy the answer *verbatim* from a source; do not paraphrase.")
    lines.append("- If the answer is a location, include the full span as written (e.g., 'Fujioka, Gunma').")
    if yn:
        lines.append("- If the question is yes/no, output exactly 'yes' or 'no'.")
    lines.append("- Only say 'unknown' if NONE of the sources contain enough information.")
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
    lines.append("Answer (short):" if not yn else "Answer (yes/no only):")
    return "\n".join(lines)

# -------- API: /answer --------
@app.get("/answer", response_model=AnswerResponse)
def answer(
    q: str = Query(..., description="Question"),
    k: int = Query(4, ge=1, le=10, description="Top-k contexts"),
    model: str = Query("qwen3:8b", description="Ollama model name"),
    temperature: float = Query(0.2, ge=0.0, le=1.0),
    max_tokens: int = Query(80, ge=16, le=512),
):
    # Validate model availability
    available = set(list_ollama_models())
    if available and model not in available:
        raise HTTPException(status_code=400, detail={
            "error": f"Model not found in Ollama: {model}",
            "available": sorted(list(available))[:50],
            "hint": "Use an installed model (e.g., 'gemma3:1b', 'gemma3:4b', 'qwen3:4b', 'deepseek-r1:8b')."
        })

    # Multi-query retrieval (question + entities), but still return only top-k contexts
    search_hits = search_multi(q, k=k, k_per_entity=3, max_pool=24)
    if not search_hits:
        raise HTTPException(status_code=404, detail="No contexts found")

    prompt = build_prompt(q, search_hits)
    t0 = time.time()
    ans = call_ollama(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    ms = int((time.time() - t0) * 1000)

    if not ans.strip():
        ans = "unknown"

    return AnswerResponse(answer=ans, hits=search_hits, latency_ms=ms)
