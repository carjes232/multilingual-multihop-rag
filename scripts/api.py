#!/usr/bin/env python3
"""
FastAPI app exposing /search and /answer (RAG) with robust Ollama calls
and entity-aware multi-query retrieval.

Run:
  uvicorn scripts.api:app --reload --port 8000
"""
import json
import logging
import os
import time
import unicodedata
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import regex as re
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

try:
    from . import embedder as emb
    from . import retriever as retr
except ImportError:
    import embedder as emb
    import retriever as retr

# Prometheus metrics
try:
    SEARCH_REQUESTS = Counter('search_requests_total', 'Total number of search requests')
    ANSWER_REQUESTS = Counter('answer_requests_total', 'Total number of answer requests')
    SEARCH_LATENCY = Histogram('search_latency_seconds', 'Latency of search requests in seconds')
    ANSWER_LATENCY = Histogram('answer_latency_seconds', 'Latency of answer requests in seconds')
except ValueError:
    from prometheus_client import REGISTRY
    SEARCH_REQUESTS = REGISTRY._names_to_collectors['search_requests_total']
    ANSWER_REQUESTS = REGISTRY._names_to_collectors['answer_requests_total']
    SEARCH_LATENCY = REGISTRY._names_to_collectors['search_latency_seconds']
    ANSWER_LATENCY = REGISTRY._names_to_collectors['answer_latency_seconds']


# -------- Config --------
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
API_TOKEN = os.getenv("API_TOKEN", None)
RATE_LIMIT = os.getenv("RATE_LIMIT", "5/minute")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1024"))

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_GEN_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"

# -------- App state --------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rag_api")

app = FastAPI(title="RAG Search API")

class TokenBearer(HTTPBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        credentials: Optional[HTTPAuthorizationCredentials] = await super().__call__(request)
        if credentials:
            if not API_TOKEN or credentials.credentials != API_TOKEN:
                raise HTTPException(status_code=403, detail="Invalid or missing token")
            return credentials.credentials
        if API_TOKEN:
             raise HTTPException(status_code=403, detail="Invalid or missing token")
        return None


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_request_size(request: Request):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        raise HTTPException(status_code=413, detail="Request size exceeds the limit")


token_bearer = TokenBearer()
_model = None
_retr: Optional[retr.PgVectorRetriever | retr.PineconeRetriever] = None


# -------- Utilities --------
def vec_to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def ensure_model():
    global _model
    if emb.get_backend() == "local" and _model is None:
        try:
            _model = emb._ensure_local_model()
            import torch
            dev = getattr(_model, "device", "cpu")
            cuda_ok = getattr(torch, "cuda", None) and torch.cuda.is_available()
            model_name = emb.get_model_name()
            logger.info("Model loaded via embedder: %s | CUDA: %s | Device: %s", model_name, bool(cuda_ok), dev)
        except Exception as e:
            _model = None
            logger.warning("Local embedder model not loaded: %s", e)
    return _model


def ensure_retriever():
    global _retr
    if _retr is None:
        _retr = retr.make_retriever()
        logger.info("Retriever backend: %s", retr.get_backend())
        if _retr:
            try:
                _retr.search(_model, ["test"], 1)
                logger.info("Retriever pre-warmed successfully.")
            except Exception as e:
                logger.warning("Failed to pre-warm retriever: %s", e)
    return _retr


def is_yesno_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return q.startswith(
        (
            "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
            "should ", "has ", "have ", "had ", "will ", "would ",
        )
    )


# --- entity extraction for multi-query retrieval ---
_QUOTED = re.compile(r'["“”](.+?)["“”]')
_UP = r"\p{Lu}"  # Any uppercase letter (Unicode)
_TAIL = r"\p{Ll}*" # Any lowercase letter (Unicode)

# Define the content pattern separately
_TITLE_SPAN_CONTENT = rf"{_UP}{_TAIL}(?:\s+{_UP}{_TAIL})+"

# Non-overlapping, word-boundary-bounded title spans (>= 2 capitalized words).
# Boundaries: start-of-string or non-letter before; end-of-string or non-letter after.
_TITLE_SPAN = re.compile(
    rf"(?:(?<=^)|(?<=[^\p{{L}}]))"
    rf"({_TITLE_SPAN_CONTENT})"
    rf"(?=$|[^\p{{L}}])",
    flags=re.UNICODE,
)

_NUM_TITLE = re.compile(
    rf"\b(\d{{1,4}}(?:st|nd|rd|th)?\s+[{_UP}]{_TAIL}(?:\s+[{_UP}]{_TAIL})*)\b",
    flags=re.UNICODE,
)

_CAP_STOP = {
    "A", "An", "The", "Is", "Are", "Was", "Were", "Do", "Does", "Did",
    "Has", "Have", "Had", "Will", "Would", "Should", "Could", "Can",
    "May", "Might", "Must", "In", "On", "At", "Of", "For", "To", "From", "By",
}


def extract_entities(q: str) -> List[str]:
    s = q or ""
    ents: set[str] = set()

    # quoted phrases
    for m in _QUOTED.finditer(s):
        phrase = m.group(1).strip()
        if phrase:
            ents.add(phrase)

    # numeric-leading titles (e.g., "21st Century Fox")
    for m in _NUM_TITLE.finditer(s):
        t = m.group(1).strip()
        if t:
            ents.add(t)

    # capitalized multi-word spans
    for m in _TITLE_SPAN.finditer(s):
        span = m.group(1).strip()
        parts = span.split()
        # Drop leading capitalized stop-words (e.g., "The", "Did") but keep inner title spans
        while len(parts) >= 2 and parts[0] in _CAP_STOP:
            parts = parts[1:]
        if len(parts) >= 2:
            cleaned = " ".join(parts)
            if cleaned:
                ents.add(cleaned)

    # dedup, prefer longer strings first
    out, seen = [], set()
    for e in sorted(list(ents), key=len, reverse=True):
        k = e.lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
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
    citations: List[int]
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


# -------- Public /search (single-query, unchanged) --------
@app.get("/search", response_model=List[SearchHit])
def search(q: str = Query(...), k: int = Query(5, ge=1, le=50)):
    SEARCH_REQUESTS.inc()
    start_time = time.time()
    model = ensure_model()
    r = ensure_retriever()
    batches = r.search(model, [q], k)  # retriever embeds via env-gated embedder
    hits = batches[0] if batches else []
    SEARCH_LATENCY.observe(time.time() - start_time)
    return [SearchHit(id=h.id, doc_id=h.doc_id, title=h.title, text=h.text, score=h.score) for h in hits]


# -------- Internal: multi-query retrieval --------
def search_multi(question: str, k: int, k_per_entity: int = 3, max_pool: int = 24) -> List[SearchHit]:
    """
    Retrieve by:
      1) the full question (k),
      2) each detected entity (k_per_entity),
    merge by id keeping the best score, re-rank, then return ONLY top-k.
    """
    model = ensure_model()
    r = ensure_retriever()
    queries: List[tuple[str, int]] = [(question, max(2, k))]  # ensure at least 2 from the full question
    # entity queries (cap to avoid over-encoding)
    ents = extract_entities(question)[:6]
    for e in ents:
        queries.append((e, k_per_entity))

    # run backend per-query with its desired k
    batches: List[List[retr.Hit]] = []
    for qtxt, kk in queries:
        res = r.search(model, [qtxt], kk)
        batches.extend(res)

    pooled = retr.pool_and_rerank(batches, top_k=k, max_pool=max_pool)
    final_rows = pooled[:k]
    return [SearchHit(id=h.id, doc_id=h.doc_id, title=h.title, text=h.text, score=h.score) for h in final_rows]


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
    if m:
        return m.group(1).lower()

    # strict JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "final" in obj:
            ans = str(obj["final"]).strip()
            if ans:
                return ans
    except Exception:
        pass

    # near-JSON "final: ..."
    m2 = re.search(r'(?is)\bfinal\b\s*:\s*["\']?([^\n\r"\'}]*)', t)
    if m2:
        cand = m2.group(1).strip()
        if cand:
            return cand

    # markers
    for pat in (
        r"(?is)\bfinal answer\b\s*:\s*(.+)",
        r"(?is)\banswer\s*\(short\)\s*:\s*(.+)",
        r"(?is)\banswer\b\s*:\s*(.+)",
    ):
        mm = re.search(pat, t)
        if mm:
            words = re.split(r"\s+", re.sub(r"\s+", " ", mm.group(1)).strip())
            return " ".join(words[:max_words]).strip()

    # fallback: last non-empty line, else first words
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        words = re.split(r"\s+", lines[-1])
        out = " ".join(words[:max_words]).strip()
        if out and not out.lower().startswith("<think"):
            return out
    return " ".join(re.split(r"\s+", t)[:max_words]).strip()


def _normalize_for_match(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def compute_citations(answer: str, hits: List[SearchHit]) -> List[int]:
    """Return 1-based indices of sources that contain the answer span.
    Normalizes diacritics/case/spacing. Ignores very short answers (<3 chars) except yes/no.
    Deduplicates and caps to 3.
    """
    ans_raw = (answer or "").strip()
    ans_norm = _normalize_for_match(ans_raw)
    if not ans_norm or ans_norm in {"yes", "no", "unknown"}:
        return []
    if len(ans_norm) < 3:
        return []
    out: list[int] = []
    for i, h in enumerate(hits, 1):
        try:
            text_norm = _normalize_for_match(h.text or "")
            if ans_norm and ans_norm in text_norm:
                out.append(i)
        except Exception:
            continue
        if len(out) >= 3:
            break
    # Dedup while keeping order
    seen: set[int] = set()
    uniq = []
    for idx in out:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)
    return uniq


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



@lru_cache(maxsize=None)


def list_ollama_models(timeout: int = 5) -> list[str]:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        if r.status_code != 200:
    
    
            return []
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
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










        return (data.get("message") or {}).get("content", "") or data.get("response", "") or ""
    return data.get("response", "") or ""


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
    sys_json = (
        'Return ONLY a JSON object {"final": "<short>"}. '
        'Do NOT include <think> or explanations. If unsure, return "unknown".'
    )

    trials = [
        (
            "chat_schema",
            {
                "model": model,
                "messages": [{"role": "system", "content": sys_json}, {"role": "user", "content": prompt}],
                "stream": False,
                "format": schema,
                "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
            },
            _chat,
        ),
        (
            "chat_json",
            {
                "model": model,
                "messages": [{"role": "system", "content": sys_json}, {"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
            },
            _chat,
        ),
        (
            "chat_plain",
            {
                "model": model,
                "messages": [{"role": "system", "content": sys_plain}, {"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
            },
            _chat,
        ),
        (
            "gen_json",
            {
                "model": model,
                "prompt": prompt + '\n\nReturn ONLY JSON: {"final": "<short>"}',
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
            },
            _generate,
        ),
        (
            "gen_plain",
            {
                "model": model,
                "prompt": prompt + "\n\nReturn only the short final answer. Do NOT include <think>.",
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
            },
            _generate,
        ),
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
@app.post("/answer", response_model=AnswerResponse, dependencies=[Depends(token_bearer)])
def answer(
    q: str = Query(..., description="Question"),
    k: int = Query(4, ge=1, le=10, description="Top-k contexts"),
    model: str = Query("qwen3:8b", description="Ollama model name"),
    temperature: float = Query(0.2, ge=0.0, le=1.0),
    max_tokens: int = Query(80, ge=16, le=512),
):
    ANSWER_REQUESTS.inc()
    start_time = time.time()
    # Validate model availability
    available = set(list_ollama_models())
    if available and model not in available:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Model not found in Ollama: {model}",
                "available": sorted(list(available))[:50],
                "hint": "Use an installed model (e.g., 'gemma3:1b', 'gemma3:4b', 'qwen3:4b', 'deepseek-r1:8b').",
            },
        )

    # Multi-query retrieval (question + entities), but still return only top-k contexts
    search_hits = search_multi(q, k=k, k_per_entity=3, max_pool=24)


    if not search_hits:
        raise HTTPException(status_code=404, detail="No contexts found")

    prompt = build_prompt(q, search_hits)
    # Removed unused variable t0
    ans = call_ollama(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    ms = int((time.time() - start_time) * 1000)
    ANSWER_LATENCY.observe(time.time() - start_time)

    # Abstention logic: if no informative span is found in any source and top score is weak.
    top_score = max((h.score for h in search_hits), default=0.0)
    yn = is_yesno_question(q)
    cits = compute_citations(ans, search_hits)
    min_top = float(os.getenv("ANSWER_MIN_TOP_SCORE", "0.30"))
    if (not yn) and (not cits) and (top_score < min_top):
        ans = "unknown"
        cits = []

    if not ans.strip():


        ans = "unknown"

    return AnswerResponse(answer=ans, hits=search_hits, citations=cits, latency_ms=ms)

# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():


    CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'
    from prometheus_client import generate_latest as prom_generate_latest

    def generate_latest():
        return prom_generate_latest()

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
