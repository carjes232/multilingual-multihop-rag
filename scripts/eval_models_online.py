#!/usr/bin/env python3
"""
Online multi-model evaluation via OpenRouter: RAG vs No-RAG.

* Compares short-answer accuracy (EM / F1) and latency
  across any OpenRouter-served models.
* Uses your local retriever (`search_multi`) + prompt builder (`build_prompt`).
* Saves CSV + five PNG plots under --out-dir (default: runtime/evals_multi/).

Env (.env or exported):
    OPENROUTER_API_KEY      (required)
    OPENROUTER_BASE_URL     (default: https://openrouter.ai/api/v1)
    OPENROUTER_HTTP_REFERER (optional)
    OPENROUTER_X_TITLE      (optional)

Example
-------
python scripts/eval_models_online.py \
  --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
  --sample 74 --k 4 \
  --models "google/gemini-2.0-flash-001,google/gemini-2.5-flash" \
  --out-dir runtime/evals_multi --write-errors
"""

from __future__ import annotations

import argparse, csv, json, os, random, re, time, unicodedata
from typing import Dict, List, Tuple, Set

import requests

# optional .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# -------- retrieval & prompt builder (local API helpers) ------------
from api import search_multi, build_prompt, is_yesno_question  # noqa

# -------- plotting (headless matplotlib) ----------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

# ===================== OpenRouter config ============================
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL",
                                     "https://openrouter.ai/api/v1")
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_X_TITLE = os.environ.get("OPENROUTER_X_TITLE", "")

# ===================== text / metric helpers ========================
_THINK_ANY  = re.compile(r"(?is)<\s*(think|thought|thinking)\b[^>]*>.*?<\s*/\s*\1\s*>")
_THINK_OPEN = re.compile(r"(?is)<\s*(think|thought|thinking)\b[^>]*>.*\Z")
_CODEFENCE  = re.compile(r"(?is)```(?:json|txt|markdown)?\s*(.*?)\s*```")

YES_SET = {"yes", "yep", "yeah", "true", "correct", "affirmative"}
NO_SET  = {"no", "nope", "false", "incorrect", "negative"}

BAD_NULLS = {
    "unknown", "n/a", "na", "none", "null",
    "cannot answer", "can't answer", "no answer",
    "unsure", "not sure", "idk"
}

ABBREV = {
    "nyc": "new york city",
    "u.s.": "united states",
    "u.s": "united states",
    "usa": "united states",
    "u.k.": "united kingdom",
    "u.k": "united kingdom",
}

_ORDINALS = {
    "1st": "1", "2nd": "2", "3rd": "3",
    **{f"{i}th": str(i) for i in range(4, 101)}
}

def _strip_diacritics(s: str) -> str:
    n = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in n if not unicodedata.combining(ch))

def _apply_abbrev(s: str) -> str:
    def repl(m):
        tok = m.group(0).lower()
        return ABBREV.get(tok, tok)
    return re.sub(r"\b[a-z.]{2,}\b", repl, s, flags=re.I)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = _strip_diacritics(s)
    s = _apply_abbrev(s)
    s = (s.replace("–", "-")
           .replace("—", "-")
           .replace("’", "'")
           .lower())
    # collapse possessive
    s = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", s)
    # ordinals → cardinals
    for k, v in _ORDINALS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    # 1999-2000 → 1999 to 2000
    s = re.sub(r"\b(\d{4})\s*-\s*(\d{2,4})\b", r"\1 to \2", s)
    # strip leading "from "
    s = re.sub(r"^\s*from\s+", "", s)
    # unify yes/no variants
    s = re.sub(r"\b(?:yeah|yep|yup|affirmative|true|correct)\b", " yes ", s)
    s = re.sub(r"\b(?:nope|negative|false|incorrect)\b", " no ", s)
    # drop punctuation
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    # articles
    s = re.sub(r"\b(?:a|an|the)\b", " ", s)
    # squeeze spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def f1_score(pred: str, gold: str) -> float:
    pt, gt = normalize_text(pred).split(), normalize_text(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0
    common, overlap = {}, 0
    for t in pt:
        common[t] = common.get(t, 0) + 1
    for t in gt:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0: return 0.0
    p, r = overlap / len(pt), overlap / len(gt)
    return 2 * p * r / (p + r)

def _name_like_equiv(p: str, g: str) -> bool:
    """
    First+Surname match with optional initials equivalence.
    """
    pt = [t for t in p.split() if t not in {"jr", "sr"}]
    gt = [t for t in g.split() if t not in {"jr", "sr"}]
    if not pt or not gt: return False
    if len(pt) > 5 or len(gt) > 5: return False
    # If both have a surname token, require equality
    if pt[-1].isalpha() and gt[-1].isalpha() and pt[-1] == gt[-1]:
        fpt, fgt = pt[0], gt[0]
        if fpt == fgt: return True
        if (len(fpt) == 1 and fgt.startswith(fpt)) or (len(fgt) == 1 and fpt.startswith(fgt)):
            return True
    # If prediction ends with an initial but prefix matches gold's first+middle
    if len(pt) >= 2 and len(gt) >= 2 and len(pt[-1]) == 1:
        if pt[0] == gt[0] and gt[1].startswith(pt[1]):  # "Henry J" vs "Henry J Kaiser"
            return True
    return False

_YEAR_RX = re.compile(r'\b(1[5-9]\d{2}|20\d{2})\b')

def _years(s: str) -> Set[str]:
    return set(_YEAR_RX.findall(s or ""))

def exact_match(pred: str, gold: str, *, em_mode: str = "lenient") -> bool:
    """
    Forgiving EM:
      * substring
      * yes/no synonyms
      * initials / name-like equivalence
      * token-subset (gold ⊆ prediction)
      * (lenient) same-year(s) match ignoring month/day
      * near-perfect F1
    """
    p = normalize_text(pred)
    g = normalize_text(gold)

    if p == g or (not p and not g): return True
    if g and g in p: return True
    if (p in YES_SET and g in YES_SET) or (p in NO_SET and g in NO_SET): return True
    if _name_like_equiv(p, g): return True

    ps, gs = set(p.split()), set(g.split())
    if gs and gs.issubset(ps): return True  # gold tokens all present

    if em_mode == "lenient":
        yp, yg = _years(p), _years(g)
        if yp and yg and yp == yg:  # e.g., "1922" vs "October 1922"
            return True

    if f1_score(pred, gold) >= 0.95: return True
    return False

def percentile(ms: List[int], p: float) -> int:
    if not ms: return 0
    xs = sorted(ms)
    idx = round((p / 100) * (len(xs) - 1))
    return xs[max(0, min(len(xs) - 1, idx))]

def median(nums: List[float]) -> float:
    if not nums: return 0.0
    xs = sorted(nums)
    n = len(xs)
    m = n // 2
    return xs[m] if n % 2 else 0.5 * (xs[m - 1] + xs[m])

def _strip_think(s: str) -> str:
    if not s: return ""
    txt = _THINK_ANY.sub("", s)
    txt = _THINK_OPEN.sub("", txt)
    mcf = _CODEFENCE.search(txt)
    if mcf: txt = mcf.group(1)
    return txt.strip()

# very loose JSON "final:" finder (works even if model never closes quotes/braces)
_FINAL_ANY = re.compile(r'(?is)["\']?\s*final\s*["\']?\s*:\s*["\']?\s*([^\n\r"}]+)')

def _salvage_from_text(t: str) -> str:
    # Try range like 1986-2013 or 1986 to 2013
    m = re.search(r'\b(\d{3,4})\s*(?:-|to|–|—)\s*(\d{2,4})\b', t)
    if m: return f"{m.group(1)}-{m.group(2)}"
    # Try Month YYYY
    m = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}\b', t, re.I)
    if m:
        span = t[m.start():m.end()]
        return re.sub(r"\s+", " ", span.strip())
    # Single year
    m = _YEAR_RX.search(t)
    if m: return m.group(1)
    # Proper-looking short noun chunk
    m = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b', t)
    if m: return m.group(1)
    return ""

def _post_prune(cand: str) -> str:
    cand = re.split(r"\s*\(", cand, 1)[0].strip()
    cand = re.split(r"\s+\b(?:is|are|was|were)\b\s+", cand, 1)[0].strip()
    return cand

def _expand_initials_from_raw(cand: str, raw: str) -> str:
    """
    If cand looks like 'First J' and raw contains 'First J. Surname',
    return the completed name; else return cand unchanged.
    """
    if not cand or not raw: return cand
    m = re.match(r'^\s*([A-Z][a-z]+)\s+([A-Z])\.?\s*$', cand)
    if not m: return cand
    first, init = m.group(1), m.group(2)
    rx = re.compile(rf'\b{re.escape(first)}\s+{re.escape(init)}\.?\s+([A-Z][a-z]+)\b')
    m2 = rx.search(raw)
    if m2:
        surname = m2.group(1)
        return f"{first} {init}. {surname}"
    return cand

def _shorten_final(s: str, allow_unknown_salvage: bool = True) -> str:
    """
    Extract a short answer from a model reply, robust to:
      * dangling <think> / code fences
      * malformed JSON (no closing quote/brace)
      * extra prose before/after
      * trailing 'is/are/was/were...' definitions
      * parenthetical tails
    """
    if not s: return ""

    t = _strip_think(s)

    # fast-path yes/no
    m_yn = re.match(r"\s*(yes|no)\b", t, flags=re.I)
    if m_yn: return m_yn.group(1).lower()

    # strict JSON up to first }
    j_end = t.find("}") + 1
    j_try = t[:j_end] if j_end > 0 else t
    try:
        obj = json.loads(j_try)
        if isinstance(obj, dict) and "final" in obj:
            cand = str(obj["final"]).strip()
            cand = _post_prune(cand)
            cand = _expand_initials_from_raw(cand, t)
            if cand.lower() in BAD_NULLS and allow_unknown_salvage:
                alt = _salvage_from_text(t)
                return alt or "unknown"
            return cand or "unknown"
    except Exception:
        pass

    # loose "final:"
    m = _FINAL_ANY.search(t)
    if m:
        cand = _post_prune(m.group(1).strip(' "\''))
        cand = _expand_initials_from_raw(cand, t)
        if cand.lower() in BAD_NULLS and allow_unknown_salvage:
            alt = _salvage_from_text(t)
            return alt or "unknown"
        return cand or "unknown"

    # heuristic: first sentence → prune → first 6 words
    sent = re.split(r"[.\n]", t, 1)[0]
    cand = _post_prune(sent.strip())
    words = re.split(r"\s+", cand)
    res = " ".join(words[:6]).strip()
    res = _expand_initials_from_raw(res, t)
    if res.lower() in BAD_NULLS and allow_unknown_salvage:
        alt = _salvage_from_text(t)
        return alt or "unknown"
    return res or "unknown"

# ===================== OpenRouter wrapper ===========================
def _openrouter_headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_X_TITLE:
        headers["X-Title"] = OPENROUTER_X_TITLE
    return headers

def call_openrouter_short(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int = 120,
    yn: bool = False,
) -> Tuple[str, int]:
    """
    Return (short_answer, latency_ms). Try several prompt styles:
      1) JSON with "final", must guess, no 'unknown'
      2) Plain short with same constraints
      3) (if yn) force "yes" or "no" only
    """
    base_sys_json = (
        'Return ONLY JSON like {"final":"<short>"} . '
        "Give your best short answer even if unsure. "
        "Do NOT output placeholders like 'unknown', 'n/a', or 'unsure'."
    )
    base_sys_plain = (
        "Answer with ONLY the short final answer. "
        "Give your best short answer even if unsure. "
        "Do NOT output placeholders like 'unknown', 'n/a', or 'unsure'."
    )
    yn_sys_plain = (
        "Answer with ONLY one word: yes or no. "
        "If unsure, choose the most likely. Do NOT output anything else."
    )

    trials = [
        {"model": model, "messages": [{"role": "system", "content": base_sys_json}] + messages,
         "temperature": max(0.0, temperature), "max_tokens": max(16, max_tokens)},
        {"model": model, "messages": [{"role": "system", "content": base_sys_plain}] + messages,
         "temperature": max(0.0, temperature), "max_tokens": max(16, max_tokens)},
    ]
    if yn:
        trials.append(
            {"model": model, "messages": [{"role": "system", "content": yn_sys_plain}] + messages,
             "temperature": 0.0, "max_tokens": 4}
        )

    headers = _openrouter_headers()
    last_content = ""
    start = time.time()

    for payload in trials:
        try:
            r = requests.post(OPENROUTER_CHAT_URL,
                              headers=headers,
                              json=payload,
                              timeout=timeout)
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue

        data = r.json()
        content = ((data or {}).get("choices") or [{}])[0].get("message", {}).get("content", "")
        content = (content or "").strip()
        last_content = content or last_content
        short = _shorten_final(content, allow_unknown_salvage=True)
        if short and not short.lower().startswith("<think"):
            return short, int((time.time() - start) * 1000)

    return (last_content or "unknown"), int((time.time() - start) * 1000)

# ===================== plotting helpers ============================
def bar_chart(x, a, b, la, lb, title, out, ylim=None):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    idx = range(len(x))
    w = 0.38
    ax.bar([i - w / 2 for i in idx], a, w, label=la)
    ax.bar([i + w / 2 for i in idx], b, w, label=lb)
    ax.set_xticks(list(range(len(x))))
    ax.set_xticklabels(x, rotation=15, ha="right")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def histogram(values, bins, title, out):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    ax.hist(values, bins=bins, color="C0", alpha=0.8)
    mu = (sum(values) / len(values)) if values else 0.0
    med = median(values)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.2, label="Zero")
    ax.axvline(mu, color="C1", linewidth=1.5, label=f"Mean {mu:.3f}")
    ax.axvline(med, color="C2", linestyle="-.", linewidth=1.5, label=f"Median {med:.3f}")
    ax.set_title(title)
    ax.set_xlabel("F1 gain (RAG − No-RAG)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

# ===================== main ========================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Online OpenRouter eval (RAG vs No-RAG).")
    ap.add_argument("--file", default="runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl")
    ap.add_argument("--out-dir", default="runtime/evals_multi")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument(
        "--models",
        default="google/gemini-2.0-flash-001,google/gemini-2.5-flash",
        help="comma-separated OpenRouter model IDs",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--norag-temperature", type=float, default=0.0)
    ap.add_argument("--norag-max-tokens", type=int, default=48)
    ap.add_argument("--em-mode", choices=["strict", "lenient"], default="lenient",
                    help="lenient counts EM true for matching year(s) and clear initial→surname equivalence.")
    ap.add_argument("--write-errors", action="store_true")
    args = ap.parse_args()

    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set (env or .env).")

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # ---------- load & sample ----------
    pool: List[Tuple[str, str]] = []
    with open(args.file, encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                q, a = ex.get("question"), ex.get("answer")
                if q and a is not None:
                    pool.append((q, a))
            except json.JSONDecodeError:
                continue

    if not pool:
        raise SystemExit("No examples found in file.")

    batch = random.sample(pool, min(args.sample, len(pool)))

    # ---------- CSV summary ----------
    summary_csv = os.path.join(args.out_dir, "model_summaries_online.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow([
            "model", "n",
            "em_rag", "f1_rag", "p50_rag_ms", "p95_rag_ms",
            "em_norag", "f1_norag", "p50_norag_ms", "p95_norag_ms",
            "em_rag_only", "em_norag_only", "em_both_true", "em_both_false",
            "f1_gain_avg", "f1_gain_median", "f1_gain_pos", "f1_gain_neg", "f1_gain_zero",
        ])

        model_names: List[str] = []
        em_rag_list: List[float] = []
        em_nr_list: List[float] = []
        f1_rag_list: List[float] = []
        f1_nr_list: List[float] = []
        p50_rag_list: List[int] = []
        p50_nr_list: List[int] = []
        em_rag_only_list: List[int] = []
        em_nr_only_list: List[int] = []
        em_both_true_list: List[int] = []
        em_both_false_list: List[int] = []
        all_gains: List[float] = []
        per_model_gains: Dict[str, List[float]] = {}
        all_errors_by_model: Dict[str, List[dict]] = {}

        # iterate requested models
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            n = rag_em = nr_em = 0
            rag_f1: List[float] = []
            nr_f1: List[float] = []
            rag_lat: List[int] = []
            nr_lat: List[int] = []
            br = bn = bt = bf = 0
            gains: List[float] = []
            error_entries: List[dict] = []

            for q, gold in batch:
                # ---------- build prompt with retrieval ----------
                try:
                    hits = search_multi(q, k=args.k, k_per_entity=3, max_pool=24)
                except Exception:
                    hits = []
                prompt = build_prompt(q, hits) if hits else q

                yn = is_yesno_question(q)

                # ---------- WITH RAG ----------
                pr, msr = call_openrouter_short(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    yn=yn,
                )
                pr_raw = pr
                pr = _shorten_final(pr)

                # ---------- NO-RAG ----------
                pn, msn = call_openrouter_short(
                    model=model,
                    messages=[{"role": "user", "content": q}],
                    temperature=args.norag_temperature,
                    max_tokens=args.norag_max_tokens,
                    yn=yn,
                )
                pn_raw = pn
                pn = _shorten_final(pn)

                # ---------- grading ----------
                if yn:
                    er = normalize_text(pr)[:3] == normalize_text(gold)[:3]
                    en = normalize_text(pn)[:3] == normalize_text(gold)[:3]
                else:
                    er = exact_match(pr, gold, em_mode=args.em_mode)
                    en = exact_match(pn, gold, em_mode=args.em_mode)

                fr = f1_score(pr, gold)
                fn = f1_score(pn, gold)

                rag_em += int(er)
                nr_em += int(en)
                rag_f1.append(fr)
                nr_f1.append(fn)
                rag_lat.append(msr)
                nr_lat.append(msn)
                gains.append(fr - fn)

                # EM breakdown counts
                if er and en: bt += 1
                elif er and not en: br += 1
                elif en and not er: bn += 1
                else: bf += 1

                # wrong cases record
                if not er or not en:
                    which = "both" if (not er and not en) else ("rag" if not er else "norag")
                    error_entries.append({
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
                    })

                n += 1

            # ---------- aggregates ----------
            em_r = rag_em / n
            em_n = nr_em / n
            f1_r = sum(rag_f1) / n
            f1_n = sum(nr_f1) / n
            p50_r = percentile(rag_lat, 50)
            p95_r = percentile(rag_lat, 95)
            p50_n = percentile(nr_lat, 50)
            p95_n = percentile(nr_lat, 95)

            pos = sum(1 for g_ in gains if g_ > 0)
            neg = sum(1 for g_ in gains if g_ < 0)
            zer = gains.count(0)

            model_names.append(model)
            em_rag_list.append(em_r)
            em_nr_list.append(em_n)
            f1_rag_list.append(f1_r)
            f1_nr_list.append(f1_n)
            p50_rag_list.append(p50_r)
            p50_nr_list.append(p50_n)
            em_rag_only_list.append(br)
            em_nr_only_list.append(bn)
            em_both_true_list.append(bt)
            em_both_false_list.append(bf)

            per_model_gains[model] = gains[:]
            all_gains.extend(gains)

            sw.writerow([
                model, n,
                round(em_r, 3), round(f1_r, 3), p50_r, p95_r,
                round(em_n, 3), round(f1_n, 3), p50_n, p95_n,
                br, bn, bt, bf,
                round(sum(gains) / n, 3), round(median(gains), 3),
                pos, neg, zer,
            ])

            if args.write_errors:
                ep = os.path.join(args.out_dir, f"errors_online_{sanitize(model)}.jsonl")
                with open(ep, "w", encoding="utf-8") as ef:
                    for rec in error_entries:
                        ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
                all_errors_by_model[model] = error_entries

    # ---------- plots ----------
    bar_chart(model_names, em_rag_list, em_nr_list,
              "RAG", "No-RAG", "Exact-Match (online)",
              os.path.join(args.out_dir, "em_online.png"), ylim=(0, 1))
    bar_chart(model_names, f1_rag_list, f1_nr_list,
              "RAG", "No-RAG", "F1 (online)",
              os.path.join(args.out_dir, "f1_online.png"), ylim=(0, 1))
    bar_chart(model_names, p50_rag_list, p50_nr_list,
              "RAG", "No-RAG", "Latency p50 (ms, online)",
              os.path.join(args.out_dir, "latency_p50_online.png"))

    # EM breakdown stacked
    fig = plt.figure(figsize=(11, 6))
    ax = fig.gca()
    idx = list(range(len(model_names)))
    ax.bar(idx, em_both_true_list, 0.5, label="EM both true")
    ax.bar(idx, em_rag_only_list, 0.5, bottom=em_both_true_list, label="RAG only true")
    bottom2 = [a + b for a, b in zip(em_both_true_list, em_rag_only_list)]
    ax.bar(idx, em_nr_only_list, 0.5, bottom=bottom2, label="No-RAG only true")
    bottom3 = [a + b for a, b in zip(bottom2, em_nr_only_list)]
    ax.bar(idx, em_both_false_list, 0.5, bottom=bottom3, label="Both false")
    ax.set_xticks(idx)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_title("EM breakdown (counts, online)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "em_breakdown_online.png"), dpi=160)
    plt.close(fig)

    hb_bins = max(5, int(len(all_gains) ** 0.5)) if all_gains else 5
    histogram(all_gains, hb_bins,
              "F1 gain histogram (RAG − No-RAG, online)",
              os.path.join(args.out_dir, "f1_gain_hist_online.png"))

    if args.write_errors:
        with open(os.path.join(args.out_dir, "all_errors_by_model_online.json"),
                  "w", encoding="utf-8") as af:
            json.dump(all_errors_by_model, af, ensure_ascii=False, indent=2)

    print(f"[✓] Wrote summary → {summary_csv}")
    for p in ("em_online.png", "f1_online.png", "latency_p50_online.png",
              "em_breakdown_online.png", "f1_gain_hist_online.png"):
        print("   └─", os.path.abspath(os.path.join(args.out_dir, p)))

if __name__ == "__main__":
    main()
