#!/usr/bin/env python3
"""
Multi-model eval: WITH RAG (/answer API) vs NO RAG (direct Ollama).

Outputs to --out-dir (default: runtime/evals_multi):
  - model_summaries.csv
  - em.png, f1.png, latency_p50.png, em_breakdown.png, f1_gain_hist.png
  - (when --write-errors)
      errors_<model>.jsonl
      all_errors_by_model.json
      all_errors.jsonl
      all_errors_best_by_model.json         <-- NEW (best-of-two per wrong case)
      all_errors_best.jsonl                 <-- NEW (flat JSONL best-of-two)
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
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"

# ---------- text/metrics ----------
_THINK_ANY = re.compile(r"(?is)<\s*(think|thought|thinking)\b[^>]*>.*?<\s*/\s*\1\s*>")
_THINK_OPEN = re.compile(r"(?is)<\s*(think|thought|thinking)\b[^>]*>.*\Z")
_CODEFENCE = re.compile(r"(?is)```(?:json|txt|markdown)?\s*(.*?)\s*```")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("–", "-").replace("—", "-").lower()
    # unify year ranges: 1986-2013 -> 1986 to 2013
    s = re.sub(r"\b(\d{4})\s*-\s*(\d{4})\b", r"\1 to \2", s)
    # drop leading 'from ' which often precedes spans
    s = re.sub(r"^\s*from\s+", "", s)
    # map common yes/no synonyms to canonical tokens
    s = re.sub(r"\b(yeah|yep|yup|affirmative|true|correct)\b", " yes ", s)
    s = re.sub(r"\b(nope|negative|false|incorrect)\b", " no ", s)
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

def _strip_think(s: str) -> str:
    if not s:
        return ""
    txt = s
    txt = _THINK_ANY.sub("", txt)
    txt = _THINK_OPEN.sub("", txt)
    # unwrap code fences if any
    mcf = _CODEFENCE.search(txt)
    if mcf: txt = mcf.group(1)
    return txt.strip()

def _shorten_final(s: str) -> str:
    """
    Think- and JSON-tolerant final extractor.
    """
    if not s: return ""
    t = _strip_think(s)

    # yes/no as first token
    m = re.match(r"\s*(yes|no)\b", t, flags=re.I)
    if m:
        return m.group(1).lower()

    # strict JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "final" in obj:
            return str(obj["final"]).strip()
    except Exception:
        pass

    # near-JSON: final: "..."
    m2 = re.search(r'(?is)\bfinal\b\s*:\s*["\']?([^\n\r"\'}]*)', t)
    if m2:
        cand = m2.group(1).strip()
        if cand: return cand

    # look for markers
    for pat in (r'(?is)\bfinal answer\b\s*:\s*(.+)',
                r'(?is)\banswer\s*\(short\)\s*:\s*(.+)',
                r'(?is)\banswer\b\s*:\s*(.+)'):
        mm = re.search(pat, t)
        if mm:
            words = re.split(r'\s+', re.sub(r'\s+', ' ', mm.group(1)).strip())
            return " ".join(words[:6]).strip()

    # fallback: first <=6 words
    words = re.split(r"\s+", t)
    return " ".join(words[:6]).strip()

# ---------- helpers ----------
def is_yesno_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return q.startswith(("is ","are ","was ","were ","do ","does ","did ",
                         "can ","could ","should ","has ","have ","had ",
                         "will ","would "))

# ---------- calls ----------
def call_answer_api_rag(q: str, k: int, model: str, temperature: float, max_tokens: int, timeout: int = 120) -> Tuple[str,int]:
    params = dict(q=q, k=k, model=model, temperature=temperature, max_tokens=max_tokens)
    t0 = time.time()
    r = requests.get(API_URL_ANSWER, params=params, timeout=timeout)
    dt = int((time.time()-t0)*1000); r.raise_for_status()
    data = r.json()
    ans = (data.get("answer") or "").strip()
    return ans, dt

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

def list_ollama_models(timeout: int = 5) -> List[str]:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json()
        models = data.get("models") or []
        out = []
        for m in models:
            name = (m or {}).get("name")
            if isinstance(name, str):
                out.append(name)
        return out
    except requests.RequestException:
        return []

def _get_text(data: dict) -> str:
    if "message" in data:
        return (data.get("message") or {}).get("content","") or data.get("response","") or ""
    return data.get("response","") or ""

def call_ollama_no_rag_json_final(model: str, question: str, temperature: float = 0.0, max_tokens: int = 96, timeout: int = 60) -> Tuple[str,int]:
    """
    Ask the base model directly (no RAG). Try schema/json/plain.
    Always return a short final string with think stripped; tolerate truncated JSON.
    """
    yn = is_yesno_question(question)
    policy = ('Return ONLY {"final": "yes"} or {"final": "no"}.'
              if yn else 'Return ONLY {"final": "<short>"} where <short> ≤ 6 words.')
    sys_json = "Do NOT include <think>. " + policy
    schema = {"type":"object","properties":{"final":{"type":"string","minLength":1,"maxLength":80}},"required":["final"]}

    trials = [
        ("chat_schema", {
            "model": model,
            "messages": [{"role":"system","content":sys_json},{"role":"user","content":question}],
            "stream": False, "format": schema,
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _chat),
        ("chat_json", {
            "model": model,
            "messages": [{"role":"system","content":sys_json},{"role":"user","content":question}],
            "stream": False, "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _chat),
        ("chat_plain", {
            "model": model,
            "messages": [{"role":"system","content":"Answer only the short final answer. Do NOT include <think>."},
                         {"role":"user","content":question}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _chat),
        ("gen_json", {
            "model": model,
            "prompt": question + "\n\nReturn ONLY JSON: {\"final\": \"<short>\"}",
            "stream": False, "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _gen),
        ("gen_plain", {
            "model": model,
            "prompt": question + "\n\nReturn only the short final answer. Do NOT include <think>.",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "repeat_penalty": 1.1},
        }, _gen),
    ]

    last_err = ""
    t0 = time.time()
    for _, payload, func in trials:
        ok, data, err = func(payload, timeout)
        if not ok:
            last_err = err; continue
        content = _get_text(data)

        # Strict JSON first
        try:
            obj = json.loads(_strip_think(content))
            if isinstance(obj, dict) and "final" in obj:
                return str(obj["final"]).strip(), int((time.time() - t0) * 1000)
        except Exception:
            pass

        # Heuristic extraction
        short = _shorten_final(content)
        if short and not short.lower().startswith("<think"):
            return short, int((time.time() - t0) * 1000)

    # last fallback
    return "unknown", int((time.time() - t0) * 1000)

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
    ax.hist(values, bins=bins, color="C0", alpha=0.75)
    # Reference lines
    mu = (sum(values)/len(values)) if values else 0.0
    med = median(values)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.2, label="Zero (no change)")
    ax.axvline(mu, color="C1", linestyle="-", linewidth=1.5, label=f"Mean ({mu:.3f})")
    ax.axvline(med, color="C2", linestyle="-.", linewidth=1.5, label=f"Median ({med:.3f})")
    ax.set_title(title)
    ax.set_xlabel("F1 gain (RAG − No-RAG)")
    ax.set_ylabel("Count")
    ax.legend()
    # Count annotations
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    zer = sum(1 for v in values if v == 0)
    txt = f"pos: {pos}  neg: {neg}  zero: {zer}  n: {len(values)}"
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#ccc"))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)

def bar_chart_multi(x_labels, stacks, stack_labels, title, out):
    fig = plt.figure(figsize=(11,6)); ax = fig.gca()
    idx = list(range(len(x_labels))); m = len(stack_labels); w = 0.8/m if m else 0.8
    for si in range(m):
        offs = [i + (si - (m-1)/2)*w for i in idx]
        ax.bar(offs, [stacks[si][i] for i in range(len(x_labels))], w, label=stack_labels[si])
    ax.set_xticks(idx); ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_title(title); ax.grid(True, axis="y", linestyle="--", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Multi-model RAG vs No-RAG eval; compact outputs + plots.")
    ap.add_argument("--file", default="runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl")
    ap.add_argument("--out-dir", default="runtime/evals_multi")
    ap.add_argument("--sample", type=int, default=10, help="Random sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--models", default="qwen3:4b,gemma3:1b,gemma3:4b,deepseek-r1:8b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--norag-temperature", type=float, default=0.0)
    ap.add_argument("--norag-max-tokens", type=int, default=96)
    ap.add_argument("--write-errors", action="store_true", help="Write per-model JSONL of wrong answers + aggregated files (+ best-of-two)")
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
        per_model_gains: Dict[str, List[float]] = {}

        # Aggregators
        all_errors_by_model: Dict[str, List[dict]] = {}
        all_errors_best_by_model: Dict[str, List[dict]] = {}

        requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
        available = set(list_ollama_models())
        if not available:
            print("[warn] Could not fetch Ollama models from /api/tags; proceeding without availability check.")
        else:
            missing = [m for m in requested_models if m not in available]
            if missing:
                print(f"[warn] Skipping missing Ollama models: {', '.join(missing)}")
            requested_models = [m for m in requested_models if m in available]
            if not requested_models:
                print("[error] No requested models are available in Ollama.")
                return

        for model in requested_models:
            n=0; rag_em=0; rag_f1=[]; rag_lat=[]; nr_em=0; nr_f1=[]; nr_lat=[]
            br=bn=bt=bf=0; gains=[]
            error_entries: List[dict] = []
            error_entries_best: List[dict] = []

            for q,gold in batch:
                # WITH RAG
                try:
                    pr, msr = call_answer_api_rag(q=q, k=args.k, model=model, temperature=args.temperature, max_tokens=args.max_tokens)
                except requests.RequestException:
                    pr, msr = "", 0
                pr_raw = pr
                pr = _shorten_final(pr)
                er = exact_match(pr, gold); fr = f1_score(pr, gold)
                rag_em += 1 if er else 0; rag_f1.append(fr); rag_lat.append(msr)

                # NO RAG
                pn, msn = call_ollama_no_rag_json_final(model=model, question=q,
                                                        temperature=args.norag_temperature,
                                                        max_tokens=args.norag_max_tokens)
                pn_raw = pn
                pn = _shorten_final(pn)
                en = exact_match(pn, gold); fn = f1_score(pn, gold)
                nr_em += 1 if en else 0; nr_f1.append(fn); nr_lat.append(msn)

                # EM breakdown
                if er and en: bt += 1
                elif er and not en: br += 1
                elif en and not er: bn += 1
                else: bf += 1

                gains.append(fr - fn)

                # Record wrong cases (either side wrong)
                if (not er) or (not en):
                    which = "both" if (not er and not en) else ("rag" if not er else "norag")
                    rec = {
                        "question": q,
                        "gold": gold,
                        "prediction_rag_raw": pr_raw,
                        "prediction_rag": pr,
                        "prediction_norag_raw": pn_raw,
                        "prediction_norag": pn,
                        "em_rag": er,
                        "em_norag": en,
                        "f1_rag": round(fr, 3),
                        "f1_norag": round(fn, 3),
                        "latency_ms_rag": msr,
                        "latency_ms_norag": msn,
                        "which_wrong": which,
                        "model": model,
                        "norm_pred_rag": normalize_text(pr),
                        "norm_pred_norag": normalize_text(pn),
                        "norm_gold": normalize_text(gold),
                    }
                    error_entries.append(rec)

                    # ---- NEW: choose a "better answer" between RAG vs No-RAG ----
                    # Priority: EM > F1 > lower latency. If tie, prefer RAG.
                    # Build comparable tuples: (em_bool, f1_float, -latency) so max() picks best
                    rag_score = (1 if er else 0, fr, -msr)
                    nor_score = (1 if en else 0, fn, -msn)

                    if rag_score >= nor_score:
                        best_source = "rag"
                        best_answer = pr
                        best_em = er
                        best_f1 = fr
                        best_latency = msr
                    else:
                        best_source = "norag"
                        best_answer = pn
                        best_em = en
                        best_f1 = fn
                        best_latency = msn

                    best_row = {
                        "question": q,
                        "gold": gold,
                        "best_source": best_source,
                        "best_answer": best_answer,
                        "best_em": bool(best_em),
                        "best_f1": round(best_f1, 3),
                        "best_latency_ms": int(best_latency),
                        "rag": {
                            "answer": pr,
                            "em": bool(er),
                            "f1": round(fr, 3),
                            "latency_ms": int(msr),
                        },
                        "norag": {
                            "answer": pn,
                            "em": bool(en),
                            "f1": round(fn, 3),
                            "latency_ms": int(msn),
                        },
                        "model": model,
                    }
                    error_entries_best.append(best_row)

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
            per_model_gains[model] = list(gains)

            if args.write_errors and error_entries:
                err_path = os.path.join(args.out_dir, f"errors_{sanitize(model)}.jsonl")
                with open(err_path, "w", encoding="utf-8") as ef:
                    for rec in error_entries:
                        ef.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # aggregate maps
            all_errors_by_model[model] = error_entries
            all_errors_best_by_model[model] = error_entries_best

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

    # Per-model F1 gain histograms
    for model, gains in per_model_gains.items():
        mbins = max(5, int(math.sqrt(max(1, len(gains)))))
        mpng = os.path.join(args.out_dir, f"f1_gain_{sanitize(model)}.png")
        histogram(gains, mbins, f"F1 gain by question: {model}", mpng)

    # Write aggregated error files
    extra_error_paths = []
    if args.write_errors:
        all_errors_json = os.path.join(args.out_dir, "all_errors_by_model.json")
        with open(all_errors_json, "w", encoding="utf-8") as af:
            json.dump(all_errors_by_model, af, ensure_ascii=False, indent=2)
        extra_error_paths.append(all_errors_json)

        all_errors_jsonl = os.path.join(args.out_dir, "all_errors.jsonl")
        with open(all_errors_jsonl, "w", encoding="utf-8") as ff:
            for model, errs in all_errors_by_model.items():
                for rec in errs:
                    ff.write(json.dumps(rec, ensure_ascii=False) + "\n")
        extra_error_paths.append(all_errors_jsonl)

        # NEW: best-of-two grouped by model
        best_by_model_json = os.path.join(args.out_dir, "all_errors_best_by_model.json")
        with open(best_by_model_json, "w", encoding="utf-8") as bf:
            json.dump(all_errors_best_by_model, bf, ensure_ascii=False, indent=2)
        extra_error_paths.append(best_by_model_json)

        # Optional: flat JSONL for best-of-two
        best_jsonl = os.path.join(args.out_dir, "all_errors_best.jsonl")
        with open(best_jsonl, "w", encoding="utf-8") as bff:
            for model, items in all_errors_best_by_model.items():
                for rec in items:
                    bff.write(json.dumps(rec, ensure_ascii=False) + "\n")
        extra_error_paths.append(best_jsonl)

    print(f"[OK] Wrote {summary_csv}")
    per_model_pngs = [os.path.join(args.out_dir, f) for f in os.listdir(args.out_dir) if f.startswith("f1_gain_") and f.endswith(".png")]
    outputs = [em_png, f1_png, lat_png, emb_png, gain_png, *per_model_pngs]
    if args.write_errors:
        outputs.extend([os.path.join(args.out_dir, f) for f in os.listdir(args.out_dir) if f.startswith("errors_") and f.endswith(".jsonl")])
        outputs.extend(extra_error_paths)
    for p in outputs:
        print(" -", os.path.abspath(p))

if __name__ == "__main__":
    main()
