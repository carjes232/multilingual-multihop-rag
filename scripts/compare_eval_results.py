#!/usr/bin/env python3
"""
Compare local vs online eval summaries and produce CSV + plots.

Inputs (CSV):
  - Local:  runtime/evals_multi/model_summaries.csv
  - Online: runtime/evals_multi/model_summaries_online.csv

Outputs (--out-dir, default: runtime/evals_multi):
  - combined_long.csv                  (env,model,... rows)
  - comparison_summary.csv             (env-level macro/micro aggregates)
  - comparison_by_model.csv            (only for overlapping models)
  - em_rag_common.png                  (per-model Local vs Online)
  - f1_rag_common.png
  - latency_p50_rag_common.png
  - em_norag_common.png
  - f1_norag_common.png
  - latency_p50_norag_common.png
  - env_em_rag.png                     (env-level Local vs Online)
  - env_f1_rag.png
  - env_latency_p50_rag.png
  - env_em_norag.png
  - env_f1_norag.png
  - env_latency_p50_norag.png

Notes
-----
This script expects the column schema written by scripts/eval_models.py and
scripts/eval_models_online.py. Metrics are joined by exact model names when
comparing per-model; if there is no overlap, only env-level aggregates and the
combined_long.csv are produced.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLUMNS = [
    "model", "n",
    "em_rag", "f1_rag", "p50_rag_ms", "p95_rag_ms",
    "em_norag", "f1_norag", "p50_norag_ms", "p95_norag_ms",
    "em_rag_only", "em_norag_only", "em_both_true", "em_both_false",
    "f1_gain_avg", "f1_gain_median", "f1_gain_pos", "f1_gain_neg", "f1_gain_zero",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sf(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _si(v: str) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def read_summary_csv(path: str) -> Dict[str, Dict[str, float]]:
    """
    Return mapping model -> row dict with typed values.
    Unknown or missing values become 0.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as fd:
        dr = csv.DictReader(fd)
        for row in dr:
            model = row.get("model", "").strip()
            if not model:
                continue
            rec: Dict[str, float] = {}
            for k in COLUMNS:
                if k == "model":
                    continue
                v = row.get(k, "0")
                rec[k] = float(_si(v)) if k in {"n", "p50_rag_ms", "p95_rag_ms", "p50_norag_ms", "p95_norag_ms",
                                                "em_rag_only", "em_norag_only", "em_both_true", "em_both_false",
                                                "f1_gain_pos", "f1_gain_neg", "f1_gain_zero"} else _sf(v)
            out[model] = rec
    return out


def write_combined_long(local: Dict[str, Dict[str, float]],
                        online: Dict[str, Dict[str, float]],
                        out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as fd:
        wr = csv.writer(fd)
        wr.writerow(["env", *COLUMNS])
        for env, data in (("local", local), ("online", online)):
            for model, rec in sorted(data.items()):
                row = [env, model]
                for k in COLUMNS:
                    if k == "model":
                        continue
                    row.append(rec.get(k, 0))
                wr.writerow(row)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _wmean(vals: List[float], weights: List[float]) -> float:
    if not vals or not weights or len(vals) != len(weights):
        return 0.0
    wsum = sum(weights)
    if wsum <= 0:
        return 0.0
    return sum(v * w for v, w in zip(vals, weights)) / wsum


def env_summary(rows: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    models = list(rows.keys())
    ns = [rows[m].get("n", 0.0) for m in models]
    em_rag = [rows[m].get("em_rag", 0.0) for m in models]
    f1_rag = [rows[m].get("f1_rag", 0.0) for m in models]
    p50_rag = [rows[m].get("p50_rag_ms", 0.0) for m in models]
    em_nr = [rows[m].get("em_norag", 0.0) for m in models]
    f1_nr = [rows[m].get("f1_norag", 0.0) for m in models]
    p50_nr = [rows[m].get("p50_norag_ms", 0.0) for m in models]
    f1_gain = [rows[m].get("f1_gain_avg", 0.0) for m in models]

    return {
        "model_count": float(len(models)),
        "total_n": float(sum(int(n) for n in ns)),
        # macro means (per-model averages)
        "em_rag_mean": _mean(em_rag),
        "f1_rag_mean": _mean(f1_rag),
        "p50_rag_ms_mean": _mean(p50_rag),
        "em_norag_mean": _mean(em_nr),
        "f1_norag_mean": _mean(f1_nr),
        "p50_norag_ms_mean": _mean(p50_nr),
        "f1_gain_avg_mean": _mean(f1_gain),
        # weighted means by n
        "em_rag_wmean": _wmean(em_rag, ns),
        "f1_rag_wmean": _wmean(f1_rag, ns),
        "p50_rag_ms_wmean": _wmean(p50_rag, ns),  # proxy for env p50
        "em_norag_wmean": _wmean(em_nr, ns),
        "f1_norag_wmean": _wmean(f1_nr, ns),
        "p50_norag_ms_wmean": _wmean(p50_nr, ns),
        "f1_gain_avg_wmean": _wmean(f1_gain, ns),
    }


def write_env_summary(local: Dict[str, Dict[str, float]],
                      online: Dict[str, Dict[str, float]],
                      out_csv: str) -> Dict[str, Dict[str, float]]:
    s_local = env_summary(local)
    s_online = env_summary(online)
    fields = [
        "env", "model_count", "total_n",
        "em_rag_mean", "em_rag_wmean", "f1_rag_mean", "f1_rag_wmean",
        "p50_rag_ms_mean", "p50_rag_ms_wmean",
        "em_norag_mean", "em_norag_wmean", "f1_norag_mean", "f1_norag_wmean",
        "p50_norag_ms_mean", "p50_norag_ms_wmean",
        "f1_gain_avg_mean", "f1_gain_avg_wmean",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fd:
        wr = csv.writer(fd)
        wr.writerow(fields)
        wr.writerow(["local"] + [s_local[k] for k in fields[1:]])
        wr.writerow(["online"] + [s_online[k] for k in fields[1:]])
    return {"local": s_local, "online": s_online}


def overlap_by_model(local: Dict[str, Dict[str, float]],
                     online: Dict[str, Dict[str, float]]) -> List[str]:
    return sorted(set(local.keys()) & set(online.keys()))


def write_model_comparison(local: Dict[str, Dict[str, float]],
                           online: Dict[str, Dict[str, float]],
                           models: List[str],
                           out_csv: str) -> None:
    fields = [
        "model",
        "n_local", "n_online",
        "em_rag_local", "em_rag_online", "em_rag_delta",
        "f1_rag_local", "f1_rag_online", "f1_rag_delta",
        "p50_rag_ms_local", "p50_rag_ms_online", "p50_rag_ms_delta",
        "em_norag_local", "em_norag_online", "em_norag_delta",
        "f1_norag_local", "f1_norag_online", "f1_norag_delta",
        "p50_norag_ms_local", "p50_norag_ms_online", "p50_norag_ms_delta",
        "f1_gain_avg_local", "f1_gain_avg_online", "f1_gain_avg_delta",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fd:
        wr = csv.writer(fd)
        wr.writerow(fields)
        for m in models:
            l = local[m]; o = online[m]
            row = [
                m,
                int(l.get("n", 0)), int(o.get("n", 0)),
                l.get("em_rag", 0.0), o.get("em_rag", 0.0), o.get("em_rag", 0.0) - l.get("em_rag", 0.0),
                l.get("f1_rag", 0.0), o.get("f1_rag", 0.0), o.get("f1_rag", 0.0) - l.get("f1_rag", 0.0),
                int(l.get("p50_rag_ms", 0)), int(o.get("p50_rag_ms", 0)), int(o.get("p50_rag_ms", 0) - l.get("p50_rag_ms", 0)),
                l.get("em_norag", 0.0), o.get("em_norag", 0.0), o.get("em_norag", 0.0) - l.get("em_norag", 0.0),
                l.get("f1_norag", 0.0), o.get("f1_norag", 0.0), o.get("f1_norag", 0.0) - l.get("f1_norag", 0.0),
                int(l.get("p50_norag_ms", 0)), int(o.get("p50_norag_ms", 0)), int(o.get("p50_norag_ms", 0) - l.get("p50_norag_ms", 0)),
                l.get("f1_gain_avg", 0.0), o.get("f1_gain_avg", 0.0), o.get("f1_gain_avg", 0.0) - l.get("f1_gain_avg", 0.0),
            ]
            wr.writerow(row)


def bar_two_series(labels: List[str], a: List[float], b: List[float],
                   la: str, lb: str, title: str, out_path: str,
                   ylim: Tuple[float, float] | None = None) -> None:
    fig = plt.figure(figsize=(max(8, 0.6 * len(labels) + 4), 6))
    ax = fig.gca()
    idx = list(range(len(labels)))
    w = 0.4
    ax.bar([i - w/2 for i in idx], a, w, label=la)
    ax.bar([i + w/2 for i in idx], b, w, label=lb)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_common_models(local: Dict[str, Dict[str, float]],
                       online: Dict[str, Dict[str, float]],
                       models: List[str], out_dir: str) -> List[str]:
    if not models:
        return []
    # Collect series in model order
    em_rag_l = [local[m].get("em_rag", 0.0) for m in models]
    em_rag_o = [online[m].get("em_rag", 0.0) for m in models]
    f1_rag_l = [local[m].get("f1_rag", 0.0) for m in models]
    f1_rag_o = [online[m].get("f1_rag", 0.0) for m in models]
    p50_rag_l = [local[m].get("p50_rag_ms", 0.0) for m in models]
    p50_rag_o = [online[m].get("p50_rag_ms", 0.0) for m in models]
    em_nr_l = [local[m].get("em_norag", 0.0) for m in models]
    em_nr_o = [online[m].get("em_norag", 0.0) for m in models]
    f1_nr_l = [local[m].get("f1_norag", 0.0) for m in models]
    f1_nr_o = [online[m].get("f1_norag", 0.0) for m in models]
    p50_nr_l = [local[m].get("p50_norag_ms", 0.0) for m in models]
    p50_nr_o = [online[m].get("p50_norag_ms", 0.0) for m in models]

    outs = []
    out = os.path.join(out_dir, "em_rag_common.png")
    bar_two_series(models, em_rag_l, em_rag_o, "Local", "Online", "EM (RAG) — common models", out, ylim=(0, 1))
    outs.append(out)
    out = os.path.join(out_dir, "f1_rag_common.png")
    bar_two_series(models, f1_rag_l, f1_rag_o, "Local", "Online", "F1 (RAG) — common models", out, ylim=(0, 1))
    outs.append(out)
    out = os.path.join(out_dir, "latency_p50_rag_common.png")
    bar_two_series(models, p50_rag_l, p50_rag_o, "Local", "Online", "Latency p50 ms (RAG) — common models", out, ylim=None)
    outs.append(out)
    out = os.path.join(out_dir, "em_norag_common.png")
    bar_two_series(models, em_nr_l, em_nr_o, "Local", "Online", "EM (No-RAG) — common models", out, ylim=(0, 1))
    outs.append(out)
    out = os.path.join(out_dir, "f1_norag_common.png")
    bar_two_series(models, f1_nr_l, f1_nr_o, "Local", "Online", "F1 (No-RAG) — common models", out, ylim=(0, 1))
    outs.append(out)
    out = os.path.join(out_dir, "latency_p50_norag_common.png")
    bar_two_series(models, p50_nr_l, p50_nr_o, "Local", "Online", "Latency p50 ms (No-RAG) — common models", out, ylim=None)
    outs.append(out)
    return outs


def plot_env_aggregates(s_local: Dict[str, float], s_online: Dict[str, float], out_dir: str) -> List[str]:
    outs = []
    def pair(a: float, b: float) -> Tuple[List[float], List[float]]:
        return [a], [b]

    out = os.path.join(out_dir, "env_em_rag.png")
    bar_two_series(["env"], *pair(s_local["em_rag_wmean"], s_online["em_rag_wmean"]),
                   "Local", "Online", "EM (RAG) — env-weighted mean", out, ylim=(0, 1))
    outs.append(out)
    out = os.path.join(out_dir, "env_f1_rag.png")
    bar_two_series(["env"], *pair(s_local["f1_rag_wmean"], s_online["f1_rag_wmean"]),
                   "Local", "Online", "F1 (RAG) — env-weighted mean", out, ylim=(0, 1))
    outs.append(out)
    out = os.path.join(out_dir, "env_latency_p50_rag.png")
    bar_two_series(["env"], *pair(s_local["p50_rag_ms_wmean"], s_online["p50_rag_ms_wmean"]),
                   "Local", "Online", "Latency p50 ms (RAG) — weighted mean of per-model p50", out)
    outs.append(out)

    out = os.path.join(out_dir, "env_em_norag.png")
    bar_two_series(["env"], *pair(s_local["em_norag_wmean"], s_online["em_norag_wmean"]),
                   "Local", "Online", "EM (No-RAG) — env-weighted mean", out, ylim=(0, 1))
    outs.append(out)
    out = os.path.join(out_dir, "env_f1_norag.png")
    bar_two_series(["env"], *pair(s_local["f1_norag_wmean"], s_online["f1_norag_wmean"]),
                   "Local", "Online", "F1 (No-RAG) — env-weighted mean", out, ylim=(0, 1))
    outs.append(out)
    out = os.path.join(out_dir, "env_latency_p50_norag.png")
    bar_two_series(["env"], *pair(s_local["p50_norag_ms_wmean"], s_online["p50_norag_ms_wmean"]),
                   "Local", "Online", "Latency p50 ms (No-RAG) — weighted mean of per-model p50", out)
    outs.append(out)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare local vs online eval results (CSV + plots)")
    ap.add_argument("--local", default="runtime/evals_multi/model_summaries.csv",
                    help="Path to local CSV produced by scripts/eval_models.py")
    ap.add_argument("--online", default="runtime/evals_multi/model_summaries_online.csv",
                    help="Path to online CSV produced by scripts/eval_models_online.py")
    ap.add_argument("--out-dir", default="runtime/evals_multi",
                    help="Output directory for comparison CSVs and plots")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    local = read_summary_csv(args.local)
    online = read_summary_csv(args.online)

    # 1) Combined long CSV of both environments
    combined_csv = os.path.join(args.out_dir, "combined_long.csv")
    write_combined_long(local, online, combined_csv)

    # 2) Env-level summary (macro + weighted means)
    summary_csv = os.path.join(args.out_dir, "comparison_summary.csv")
    env_summaries = write_env_summary(local, online, summary_csv)

    # 3) Per-model comparison for overlapping models (if any)
    models = overlap_by_model(local, online)
    if models:
        by_model_csv = os.path.join(args.out_dir, "comparison_by_model.csv")
        write_model_comparison(local, online, models, by_model_csv)
        plot_common_models(local, online, models, args.out_dir)

    # 4) Env-level plots
    plot_env_aggregates(env_summaries["local"], env_summaries["online"], args.out_dir)

    print("[OK] Wrote:")
    print(" -", os.path.abspath(combined_csv))
    print(" -", os.path.abspath(summary_csv))
    if models:
        print(" -", os.path.abspath(os.path.join(args.out_dir, "comparison_by_model.csv")))
    for p in sorted([f for f in os.listdir(args.out_dir) if f.endswith(".png") and (
            f.endswith("_common.png") or f.startswith("env_")
        )]):
        print(" -", os.path.abspath(os.path.join(args.out_dir, p)))


if __name__ == "__main__":
    main()

