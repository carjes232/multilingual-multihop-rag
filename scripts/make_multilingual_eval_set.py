#!/usr/bin/env python3
"""
Create a 10-example multilingual JSONL from HotpotQA slice.

Reads the Hotpot validation JSONL, samples N (default 10) examples with
valid context titles, and emits a JSONL where each selected item is
expanded into three rows (EN/ES/PT):

  {"id": "<orig>-en", "lang": "en", "question": "...", "answer": "...", "context": {...}}
  {"id": "<orig>-es", "lang": "es", "question": "...", "answer": "...", "context": {...}}
  {"id": "<orig>-pt", "lang": "pt", "question": "...", "answer": "...", "context": {...}}

Questions are translated via OpenRouter (ES/PT), answers and context are
kept in English (for robust EM/F1 with entity-style answers). Requires
OPENROUTER_API_KEY in environment or .env (see .env.example).

Example:
  python scripts/make_multilingual_eval_set.py \
    --in-file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
    --out-file runtime/data/raw/hotpot/hotpot_multilingual_10.jsonl \
    --n 10 --seed 42 --model google/gemini-2.0-flash-001
"""

from __future__ import annotations

import argparse, json, os, random, time
from typing import Dict, List, Tuple

import requests

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_X_TITLE = os.environ.get("OPENROUTER_X_TITLE", "")


def _headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Put it in .env or export it.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_X_TITLE:
        headers["X-Title"] = OPENROUTER_X_TITLE
    return headers


def translate_question(q_en: str, target_lang: str, model: str, temperature: float = 0.0, timeout: int = 60) -> str:
    """Translate English question to target language using OpenRouter.
    Returns the translated question string only (no quotes or metadata).
    """
    assert target_lang in ("es", "pt"), "target_lang must be 'es' or 'pt'"
    sys = (
        "You are a translator. Translate the user's question to the target language "
        "while preserving named entities verbatim. Return ONLY the translated question, "
        "no quotes, no extra text."
    )
    target_name = {"es": "Spanish", "pt": "Portuguese"}[target_lang]
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Target language: {target_name}.\nQuestion: {q_en}"},
    ]
    payload = {"model": model, "messages": messages, "temperature": max(0.0, temperature), "max_tokens": 96}

    r = requests.post(OPENROUTER_CHAT_URL, headers=_headers(), json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:200]}")
    data = r.json()
    choices = (data or {}).get("choices") or []
    if not choices:
        return q_en
    msg = (choices[0] or {}).get("message") or {}
    content = (msg.get("content") or "").strip()
    # Strip wrapping quotes if present
    if (content.startswith("\"") and content.endswith("\"")) or (content.startswith("'") and content.endswith("'")):
        content = content[1:-1].strip()
    return content or q_en


def has_gold_titles(ex: dict) -> bool:
    ctx = ex.get("context", {})
    titles = ctx.get("title", [])
    return isinstance(titles, list) and len(titles) > 0


def main():
    ap = argparse.ArgumentParser(description="Build multilingual eval JSONL (EN/ES/PT) from Hotpot slice")
    ap.add_argument("--in-file", default="runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl")
    ap.add_argument("--out-file", default="runtime/data/raw/hotpot/hotpot_multilingual_10.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="google/gemini-2.0-flash-001", help="OpenRouter model used for translation")
    ap.add_argument("--sleep-ms", type=int, default=200, help="Sleep between API calls to be polite")
    args = ap.parse_args()

    random.seed(args.seed)
    pool: List[dict] = []
    with open(args.in_file, "r", encoding="utf-8") as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = ex.get("question")
            a = ex.get("answer")
            if not q or a is None:
                continue
            if not has_gold_titles(ex):
                continue
            ex["__row_index"] = i
            pool.append(ex)

    if not pool:
        raise SystemExit("No usable examples found in input file.")

    sample = random.sample(pool, min(args.n, len(pool)))

    out_path = args.out_file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    count_rows = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ex in sample:
            q_en = ex.get("question")
            answer = ex.get("answer")
            ctx = ex.get("context")
            base_id = ex.get("id") or f"row{ex.get('__row_index', 0)}"

            # EN row
            row_en = {
                "id": f"{base_id}-en",
                "lang": "en",
                "question": q_en,
                "answer": answer,
                "context": ctx,
            }
            out.write(json.dumps(row_en, ensure_ascii=False) + "\n")
            count_rows += 1

            # ES/PT rows via translation
            try:
                q_es = translate_question(q_en, target_lang="es", model=args.model)
                time.sleep(args.sleep_ms / 1000.0)
                q_pt = translate_question(q_en, target_lang="pt", model=args.model)
            except Exception as e:
                # If translation fails, fall back to EN to keep the sample usable
                q_es = q_en
                q_pt = q_en

            row_es = {
                "id": f"{base_id}-es",
                "lang": "es",
                "question": q_es,
                "answer": answer,
                "context": ctx,
            }
            row_pt = {
                "id": f"{base_id}-pt",
                "lang": "pt",
                "question": q_pt,
                "answer": answer,
                "context": ctx,
            }
            out.write(json.dumps(row_es, ensure_ascii=False) + "\n")
            out.write(json.dumps(row_pt, ensure_ascii=False) + "\n")
            count_rows += 2

    print(f"[OK] Wrote {out_path} with {count_rows} rows (3 per example)")


if __name__ == "__main__":
    main()

