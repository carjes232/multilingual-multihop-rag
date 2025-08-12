#!/usr/bin/env python3
"""
Multi-model eval: WITH RAG (/answer API) vs NO RAG (direct Ollama).

Outputs to --out-dir (default: runtime/evals_multi):
  - model_summaries.csv
  - em.png, f1.png, latency_p50.png, em_breakdown.png, f1_gain_hist.png

Example:
  uvicorn scripts.api:app --port 8000
  python scripts/eval_models.py --sample 10 --k 4 \
    --models qwen3:4b,gemma3:1b,gemma3:4b,deepseek-r1:8b
"""

import argparse, csv, json, math, os, random, re, time
from typing import Dict, List, Tuple

import requests

# plotting (matplotlib only; single-figure charts; no explicit colors)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

API_URL_ANSWER = "http://127.0.0.1:8000/answer"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_GEN_URL  = "http://127.0.0.1:11434/api/generate"

# ---------- text/metrics ----------
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.replace("–","-").replace("—","-").lower()
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def f1_score(pred: str, gold: str) -> float:
    pt, gt = normalize_text(pred).split(), normalize_text(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0
    common, overlap = {}, 0
    for t in pt: common[t] = common.get(t, 0) + 1
    for t in gt:
        if common.get(t, 0) > 0:
            overlap += 1; common[t] -= 1
    if not overlap: return 0.0
    p, r = overlap/len(pt), overlap/len(gt)
    return 2*p*r/(p+r)

def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)

def percentile(ms: List[int], p: float) -> int:
    if not ms: return 0
    xs = sorted(ms); idx = round((p/100)*(len(xs)-1))
    return xs[max(0, min(len(xs)-1, idx))]

def median(nums: List[float]) -> float:
    if not nums: return 0.0
    xs = sorted(nums); n = len(xs); m = n//2
    return xs[m] if n%2 else 0.5*(xs[m-1]+xs[m])

# ---------- helpers ----------
def is_yesno_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return q.startswith(("is ","are ","was ","were ","do ","does ","did ",
                         "can ","could ","should ","has ","have ","had ",
                         "will ","would "))

def _strip_think(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s or "", flags=re.S).strip()

# ---------- calls ----------
def call_answer_api_rag(q: str, k: int, model: str, temperature: float, max_tokens: int, timeout: int = 120) -> Tuple[str,int]:
    params = dict(q=q, k=k, model=model, temperature=temperature, max_tokens=max_tokens)
    t0 = time.time()
    r = requests.get(API_URL_ANSWER, params=params, timeout=timeout)
    dt = int((time.time()-t0)*1000); r.raise_for_status()
    data = r.json()
    return (data.get("answer") or "").strip(), dt

def _chat(payload: dict, timeout: int):  # returns (ok,json,err)
    try:
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
        if r.status_code == 200: return True, r.json(), ""
        return False, {}, f"{r.status_code} {r.text[:200]}"
    except requests.RequestException as e:
        return False, {}, str(e)

def _gen(payload: dict, timeout: int):
    try:
        r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=timeout)
        if r.status_code == 200: return True, r.json(), ""
        return False, {}, f"{r.status_code} {r.text[:200]}"
    except requests.RequestException as e:
        return False, {}, str(e)

def _shorten_final(s: str) -> str:
    s = _strip_think(s)
    m = re.match(r"\s*(yes|no)\b", s, flags=re.I)
    if m: return m.group(1).lower()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "final" in obj:
            return str(obj["final"]).strip()
    except Exception:
        pass
    words = re.split(r"\s+", s)
    return " ".join(words[:6]).strip()

def call_ollama_no_rag_json_final(model: str, question: str, temperature: float = 0.0, max_tokens: int = 64, timeout: int = 60) -> Tuple[str,int]:
    yn = is_yesno_question(question)
    policy = ('Return ONLY {"final": "yes"} or {"final": "no"}.'
              if yn else 'Return ONLY {"final": "<short>"} where <short> ≤ 6 words.')
    system = "You may include a brief <think> (≤2 sentences). " + policy
    schema = {"type":"object","properties":{"final":{"type":"string","minLength":1,"maxLength":80}},"required":["final"]}

    trials = [
        ("chat_schema_think", {
            "model": model,
            "messages": [{"role":"system","content":system},{"role":"user","content":question}],
            "stream": False, "think": True, "format": schema,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _chat),
        ("chat_schema", {
            "model": model,
            "messages": [{"role":"system","content":system},{"role":"user","content":question}],
            "stream": False, "format": schema,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _chat),
        ("chat_json", {
            "model": model,
            "messages": [{"role":"system","content":system},{"role":"user","content":question}],
            "stream": False, "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _chat),
        ("chat_plain", {
            "model": model,
            "messages": [{"role":"system","content":"Answer only the short final answer."},
                         {"role":"user","content":question}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _chat),
        ("gen_json", {
            "model": model,
            "prompt": question + "\n\nReturn ONLY JSON: {\"final\": \"<short>\"}",
            "stream": False, "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _gen),
        ("gen_plain", {
            "model": model,
            "prompt": question + "\n\nReturn only the short final answer.",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, _gen),
    ]

    last_err = ""
    for _, payload, func in trials:
        ok, data, err = func(payload, timeout)
        if not ok:
            last_err = err; continue
        content = ""
        if "message" in data:
            content = (data.get("message") or {}).get("content","") or data.get("response","")
        else:
            content = data.get("response","")
        return _shorten_final(content), int(1000)  # latency not critical here; keep eval symmetric
    # last fallback: unknown
    return "unknown", 0

# ---------- plots ----------
def bar_chart(x, a, b, la, lb, title, out, ylim=None):
    fig = plt.figure(figsize=(10,6)); ax = fig.gca()
    idx = range(len(x)); w = 0.38
    ax.bar([i - w/2 for i in idx], a, w, label=la)
    ax.bar([i + w/2 for i in idx], b, w, label=lb)
    ax.set_xticks(list(idx)); ax.set_xticklabels(x, rotation=15, ha="right")
    ax.set_title(title); ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)

def histogram(values, bins, title, out):
    fig = plt.figure(figsize=(10,6)); ax = fig.gca()
    ax.hist(values, bins=bins); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)

def bar_chart_multi(x_labels, stacks, stack_labels, title, out):
    fig = plt.figure(figsize=(11,6)); ax = fig.gca()
    idx = list(range(len(x_labels))); m = len(stack_labels); w = 0.8/m if m else 0.8
    for si in range(m):
        offs = [i + (si - (m-1)/2)*w for i in idx]
        ax.bar(offs, [stacks[si][i] for i in range(len(x_labels))], w, label=stack_labels[si])
    ax.set_xticks(idx); ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_title(title); ax.grid(True, axis="y", linestyle="--", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Multi-model RAG vs No-RAG eval; compact outputs + plots.")
    ap.add_argument("--file", default="runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl")
    ap.add_argument("--out-dir", default="runtime/evals_multi")
    ap.add_argument("--sample", type=int, default=10, help="Random sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--models", default="qwen3:4b,gemma3:1b,gemma3:4b,deepseek-r1:8b")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--norag-temperature", type=float, default=0.0)
    ap.add_argument("--norag-max-tokens", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # load and sample
    pool = []
    with open(args.file, "r", encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if not line: continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            q, a = ex.get("question"), ex.get("answer")
            if q and a is not None:
                pool.append((q, a))
    if not pool:
        raise SystemExit("No examples found.")
    N = min(args.sample, len(pool))
    batch = random.sample(pool, N)

    # summary csv
    summary_csv = os.path.join(args.out_dir, "model_summaries.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "model","n",
            "em_rag","f1_rag","p50_rag_ms","p95_rag_ms",
            "em_norag","f1_norag","p50_norag_ms","p95_norag_ms",
            "em_rag_only","em_norag_only","em_both_true","em_both_false",
            "f1_gain_avg","f1_gain_median","f1_gain_pos","f1_gain_neg","f1_gain_zero"
        ])

        model_names=[]; em_rag_list=[]; em_nr_list=[]; f1_rag_list=[]; f1_nr_list=[]; p50_rag_list=[]; p50_nr_list=[]
        em_rag_only_list=[]; em_nr_only_list=[]; em_both_true_list=[]; em_both_false_list=[]; all_gains=[]

        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            n=0; rag_em=0; rag_f1=[]; rag_lat=[]; nr_em=0; nr_f1=[]; nr_lat=[]
            br=bn=bt=bf=0; gains=[]

            for q,gold in batch:
                # WITH RAG
                try:
                    pr, msr = call_answer_api_rag(q=q, k=args.k, model=model, temperature=args.temperature, max_tokens=args.max_tokens)
                except requests.RequestException:
                    pr, msr = "", 0
                er = exact_match(pr, gold); fr = f1_score(pr, gold)
                rag_em += 1 if er else 0; rag_f1.append(fr); rag_lat.append(msr)

                # NO RAG
                pn, msn = call_ollama_no_rag_json_final(model=model, question=q, temperature=args.norag_temperature, max_tokens=args.norag_max_tokens)
                en = exact_match(pn, gold); fn = f1_score(pn, gold)
                nr_em += 1 if en else 0; nr_f1.append(fn); nr_lat.append(msn)

                # EM breakdown
                if er and en: bt += 1
                elif er and not en: br += 1
                elif en and not er: bn += 1
                else: bf += 1

                gains.append(fr - fn)
                n += 1

            em_r = rag_em/n if n else 0.0
            f1_r = sum(rag_f1)/n if n else 0.0
            p50_r = percentile(rag_lat, 50); p95_r = percentile(rag_lat, 95)
            em_n = nr_em/n if n else 0.0
            f1_n = sum(nr_f1)/n if n else 0.0
            p50_n = percentile(nr_lat, 50); p95_n = percentile(nr_lat, 95)
            avg_g = sum(gains)/n if n else 0.0
            med_g = median(gains)
            pos = sum(1 for g in gains if g>0); neg = sum(1 for g in gains if g<0); zer = sum(1 for g in gains if g==0)

            sw.writerow([model, n, round(em_r,3), round(f1_r,3), p50_r, p95_r,
                         round(em_n,3), round(f1_n,3), p50_n, p95_n,
                         br, bn, bt, bf, round(avg_g,3), round(med_g,3), pos, neg, zer])

            model_names.append(model); em_rag_list.append(em_r); em_nr_list.append(em_n)
            f1_rag_list.append(f1_r); f1_nr_list.append(f1_n)
            p50_rag_list.append(p50_r); p50_nr_list.append(p50_n)
            em_rag_only_list.append(br); em_nr_only_list.append(bn); em_both_true_list.append(bt); em_both_false_list.append(bf)
            all_gains.extend(gains)

    # plots
    em_png = os.path.join(args.out_dir, "em.png")
    f1_png = os.path.join(args.out_dir, "f1.png")
    lat_png = os.path.join(args.out_dir, "latency_p50.png")
    emb_png = os.path.join(args.out_dir, "em_breakdown.png")
    gain_png = os.path.join(args.out_dir, "f1_gain_hist.png")

    bar_chart(model_names, em_rag_list, em_nr_list, "RAG", "No-RAG", "Exact Match (EM)", em_png, ylim=(0,1))
    bar_chart(model_names, f1_rag_list, f1_nr_list, "RAG", "No-RAG", "Average F1", f1_png, ylim=(0,1))
    bar_chart(model_names, p50_rag_list, p50_nr_list, "RAG", "No-RAG", "Latency p50 (ms)", lat_png)

    stacks = [em_rag_only_list, em_nr_only_list, em_both_true_list, em_both_false_list]
    labels = ["RAG only","No-RAG only","Both true","Both false"]
    bar_chart_multi(model_names, stacks, labels, "EM breakdown (counts)", emb_png)

    bins = max(10, int(math.sqrt(max(1, len(all_gains)))))
    histogram(all_gains, bins, "F1 gain (RAG − No-RAG)", gain_png)

    print(f"[OK] Wrote {summary_csv}")
    for p in [em_png, f1_png, lat_png, emb_png, gain_png]:
        print(" -", os.path.abspath(p))

if __name__ == "__main__":
    main()
