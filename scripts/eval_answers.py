#!/usr/bin/env python3
"""
Evaluate answers WITH RAG (/answer API) vs NO RAG (direct Ollama).

Outputs (to --out-dir, default: runtime/evals/):
  - rag_failures.jsonl
  - rag_successes.jsonl
  - norag_failures.jsonl
  - norag_successes.jsonl
  - comparative.csv
  - summary.json
  - top_f1_gains.jsonl (top 15)
  - top_f1_losses.jsonl (top 15)

Usage:
  uvicorn scripts.api:app --port 8000
  python scripts/eval_answers.py --limit 50 --k 4 --model qwen3:8b \
    --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl
"""

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List, Tuple

import requests

API_URL_ANSWER = "http://127.0.0.1:8000/answer"   # your FastAPI RAG route
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"


# ----------------------------
# Text normalization & metrics
# ----------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # unify dashes to spaces so "1969–1974" ~ "1969 1974"
    s = s.replace("–", "-").replace("—", "-")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def percentile(ms: List[int], p: float) -> int:
    if not ms:
        return 0
    ms_sorted = sorted(ms)
    idx = round((p / 100.0) * (len(ms_sorted) - 1))
    idx = max(0, min(len(ms_sorted) - 1, idx))
    return ms_sorted[idx]


# ----------------------------
# Helpers
# ----------------------------
def is_yesno_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ",
                         "can ", "could ", "should ", "has ", "have ", "had ",
                         "will ", "would "))


# ----------------------------
# Calls: WITH RAG  (/answer)
# ----------------------------
def call_answer_api_rag(
    q: str,
    k: int,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int = 120,
) -> Tuple[str, int]:
    """
    Calls your FastAPI /answer route which already does retrieval and calls Ollama.
    Returns (prediction, latency_ms).
    """
    params = dict(q=q, k=k, model=model, temperature=temperature, max_tokens=max_tokens)
    t0 = time.time()
    r = requests.get(API_URL_ANSWER, params=params, timeout=timeout)
    dt = int((time.time() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    return (data.get("answer") or "").strip(), dt


# ----------------------------
# Calls: NO RAG (Ollama chat)
# ----------------------------
def call_ollama_no_rag_json_final(
    model: str,
    question_only: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
    timeout: int = 60,
    enable_thinking: bool = True,
) -> Tuple[str, int]:
    """
    Direct call to Ollama /api/chat with ONLY the question (no contexts).
    Uses structured outputs (JSON schema) to force a short 'final' string.
    If yes/no, instruct strictly returning 'yes' or 'no'.
    Returns (prediction, latency_ms).
    """
    yn = is_yesno_question(question_only)
    policy = (
        'Return ONLY a JSON object {"final": "<short>"} where final is exactly "yes" or "no".'
        if yn else
        'Return ONLY a JSON object {"final": "<short>"} with a SHORT span (≤6 words) that answers the question.'
    )

    system = (
        "You may include a brief <think> plan (max 2 short sentences). "
        + policy
    )
    schema = {
        "type": "object",
        "properties": {"final": {"type": "string", "minLength": 1, "maxLength": 80}},
        "required": ["final"],
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question_only},
        ],
        "stream": False,
        "think": enable_thinking,
        "format": schema,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    t0 = time.time()
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
    dt = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    data = resp.json()

    content = (data.get("message") or {}).get("content", "") or data.get("response", "")
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()

    try:
        obj = json.loads(content)
        ans = (obj.get("final") or "").strip()
    except Exception:
        m = re.search(r'\"final\"\s*:\s*\"(.*?)\"', content, flags=re.S)
        ans = m.group(1).strip() if m else content.splitlines()[0].strip()

    return ans, dt


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate WITH RAG vs NO RAG; write splits, comparative CSV and summary.")
    ap.add_argument("--file", default="runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl")
    ap.add_argument("--out-dir", default="runtime/evals")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--k", type=int, default=4, help="Top-k contexts for RAG")
    ap.add_argument("--model", default="qwen3:8b", help="Ollama model name")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=64, help="Cap for model decoding (think+answer)")
    ap.add_argument("--norag-temperature", type=float, default=0.0)
    ap.add_argument("--norag-max-tokens", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {
        "rag_fail": os.path.join(args.out_dir, "rag_failures.jsonl"),
        "rag_succ": os.path.join(args.out_dir, "rag_successes.jsonl"),
        "nr_fail":  os.path.join(args.out_dir, "norag_failures.jsonl"),
        "nr_succ":  os.path.join(args.out_dir, "norag_successes.jsonl"),
        "cmp":      os.path.join(args.out_dir, "comparative.csv"),
        "summary":  os.path.join(args.out_dir, "summary.json"),
        "gains":    os.path.join(args.out_dir, "top_f1_gains.jsonl"),
        "losses":   os.path.join(args.out_dir, "top_f1_losses.jsonl"),
    }

    files = {k: open(v, "w", encoding="utf-8", newline="") for k, v in paths.items() if k not in ("summary",)}
    cmp_writer = csv.writer(files["cmp"])
    cmp_writer.writerow([
        "question","gold",
        "pred_rag","em_rag","f1_rag","latency_ms_rag",
        "pred_norag","em_norag","f1_norag","latency_ms_norag",
        "rag_better","f1_gain",
    ])

    # Aggregates
    n = 0
    # RAG metrics
    rag_em_hits = 0
    rag_f1_sum = 0.0
    rag_latencies: List[int] = []
    # No-RAG metrics
    nr_em_hits = 0
    nr_f1_sum = 0.0
    nr_latencies: List[int] = []

    samples_printed = 0
    gains_buf: List[Tuple[float, Dict[str, object]]] = []  # (f1_gain, row_obj)

    with open(args.file, "r", encoding="utf-8") as fd:
        for line in fd:
            if args.limit and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            q = ex.get("question")
            gold = ex.get("answer")
            if not q or gold is None:
                continue

            # ---- WITH RAG ----
            try:
                pred_rag, ms_rag = call_answer_api_rag(
                    q=q, k=args.k, model=args.model,
                    temperature=args.temperature, max_tokens=args.max_tokens
                )
            except requests.RequestException as e:
                print("RAG API error:", e)
                continue

            em_rag = exact_match(pred_rag, gold)
            f1_rag = f1_score(pred_rag, gold)
            rag_em_hits += 1 if em_rag else 0
            rag_f1_sum += f1_rag
            rag_latencies.append(ms_rag)

            # ---- NO RAG ----
            try:
                pred_nr, ms_nr = call_ollama_no_rag_json_final(
                    model=args.model,
                    question_only=q,
                    temperature=args.norag_temperature,
                    max_tokens=args.norag_max_tokens,
                )
            except requests.RequestException as e:
                print("Ollama (no-RAG) error:", e)
                pred_nr, ms_nr = "", 0

            em_nr = exact_match(pred_nr, gold)
            f1_nr = f1_score(pred_nr, gold)
            nr_em_hits += 1 if em_nr else 0
            nr_f1_sum += f1_nr
            nr_latencies.append(ms_nr)

            if samples_printed < 5:
                print(f"Q: {q}\nA*: {gold}\nA_RAG : {pred_rag}\nA_NR  : {pred_nr}\n"
                      f"EM_RAG={em_rag} F1_RAG={f1_rag:.3f} t_RAG={ms_rag}ms | "
                      f"EM_NR={em_nr} F1_NR={f1_nr:.3f} t_NR={ms_nr}ms\n---")
                samples_printed += 1

            row_obj: Dict[str, object] = {
                "question": q, "gold": gold,
                "pred_rag": pred_rag, "em_rag": em_rag, "f1_rag": f1_rag, "latency_ms_rag": ms_rag,
                "pred_norag": pred_nr, "em_norag": em_nr, "f1_norag": f1_nr, "latency_ms_norag": ms_nr,
            }

            # Write per-mode splits
            files["rag_succ" if em_rag else "rag_fail"].write(json.dumps(row_obj, ensure_ascii=False) + "\n")
            files["nr_succ"  if em_nr  else "nr_fail" ].write(json.dumps(row_obj, ensure_ascii=False) + "\n")

            # Comparative CSV row
            rag_better = (f1_rag > f1_nr) or (em_rag and not em_nr)
            f1_gain = round(f1_rag - f1_nr, 6)
            cmp_writer.writerow([
                q, gold,
                pred_rag, int(em_rag), f"{f1_rag:.3f}", ms_rag,
                pred_nr,  int(em_nr),  f"{f1_nr:.3f}",  ms_nr,
                int(rag_better), f1_gain
            ])

            gains_buf.append((f1_gain, row_obj))
            n += 1

    # Close stream files (except summary)
    for k, f in files.items():
        if k != "summary":
            f.close()

    # Summaries
    rag_em = rag_em_hits / n if n else 0.0
    rag_f1 = rag_f1_sum / n if n else 0.0
    rag_p50 = percentile(rag_latencies, 50)
    rag_p95 = percentile(rag_latencies, 95)

    nr_em = nr_em_hits / n if n else 0.0
    nr_f1 = nr_f1_sum / n if n else 0.0
    nr_p50 = percentile(nr_latencies, 50)
    nr_p95 = percentile(nr_latencies, 95)

    # Win/loss/tie counts (EM)
    em_both_true = 0
    em_rag_only = 0
    em_nr_only = 0
    em_both_false = 0

    # We need to recompute these from the comparative CSV quickly
    # (or track during loop; for clarity, re-scan comparative.csv)
    with open(paths["cmp"], "r", encoding="utf-8") as cf:
        rdr = csv.DictReader(cf)
        for row in rdr:
            r = row["em_rag"] == "1"
            nr = row["em_norag"] == "1"
            if r and nr:
                em_both_true += 1
            elif r and not nr:
                em_rag_only += 1
            elif nr and not r:
                em_nr_only += 1
            else:
                em_both_false += 1

    # Top gains/losses
    gains_buf.sort(key=lambda x: x[0], reverse=True)
    top_gains = [obj for _, obj in gains_buf[:15]]
    top_losses = [obj for _, obj in gains_buf[-15:]]
    with open(paths["gains"], "w", encoding="utf-8") as fg:
        for obj in top_gains:
            fg.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(paths["losses"], "w", encoding="utf-8") as fl:
        for obj in top_losses:
            fl.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Gain distribution
    pos_gain = sum(1 for g, _ in gains_buf if g > 0)
    neg_gain = sum(1 for g, _ in gains_buf if g < 0)
    zero_gain = sum(1 for g, _ in gains_buf if g == 0)
    avg_gain = sum(g for g, _ in gains_buf) / n if n else 0.0
    med_gain = percentile([int(1000 * g) for g, _ in gains_buf], 50) / 1000.0 if n else 0.0

    summary = {
        "total": n,
        "rag": {
            "em": round(rag_em, 3),
            "f1": round(rag_f1, 3),
            "latency_ms_p50": rag_p50,
            "latency_ms_p95": rag_p95,
        },
        "no_rag": {
            "em": round(nr_em, 3),
            "f1": round(nr_f1, 3),
            "latency_ms_p50": nr_p50,
            "latency_ms_p95": nr_p95,
        },
        "em_breakdown": {
            "both_true": em_both_true,
            "rag_only": em_rag_only,
            "norag_only": em_nr_only,
            "both_false": em_both_false,
        },
        "f1_gain": {
            "count_positive": pos_gain,
            "count_negative": neg_gain,
            "count_zero": zero_gain,
            "avg_gain": round(avg_gain, 3),
            "median_gain": round(med_gain, 3),
        },
        "files": {k: os.path.abspath(v) for k, v in paths.items()},
    }
    with open(paths["summary"], "w", encoding="utf-8") as fs:
        json.dump(summary, fs, ensure_ascii=False, indent=2)

    # Console summary
    print(f"Evaluated: {n} | k={args.k} | model={args.model}")
    print(f"[WITH RAG ] EM: {rag_em:.3f}  F1: {rag_f1:.3f}  Lat p50: {rag_p50} ms  p95: {rag_p95} ms")
    print(f"[NO  RAG ] EM: {nr_em:.3f}  F1: {nr_f1:.3f}  Lat p50: {nr_p50} ms  p95: {nr_p95} ms")
    print(f"EM breakdown — RAG only wins: {em_rag_only} | No-RAG only wins: {em_nr_only} | both true: {em_both_true} | both false: {em_both_false}")
    print(f"F1 gains — +: {pos_gain}  −: {neg_gain}  0: {zero_gain} | avg gain: {avg_gain:.3f} | median gain: {med_gain:.3f}")
    print("Files written:")
    for k, v in paths.items():
        print(f" - {k:10s} -> {os.path.abspath(v)}")


if __name__ == "__main__":
    main()
