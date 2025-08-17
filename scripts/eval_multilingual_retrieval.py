#!/usr/bin/env python3
"""
Multilingual retrieval smoke test (EN/ES/PT).

Purpose: Show that cross-lingual queries retrieve overlapping contexts.
Metric: Jaccard overlap of top-k retrieved doc ids between languages.

Outputs under --out-dir (default: runtime/evals_multi/retrieval/):
  - multilingual_overlap.png  (bar chart of overlaps per query)
  - multilingual_overlap.csv  (optional via --save-csv)

Usage (defaults include a tiny built-in set of queries):
  python scripts/eval_multilingual_retrieval.py --k 5 --limit 3 \
    --out-dir runtime/evals_multi/retrieval --save-csv

You can also provide your own queries JSON (see --file format below).
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from api import search_multi


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_queries() -> List[Dict[str, str]]:
    return [
        {
            "id": "random_house_tower_location",
            "en": "Where is the Random House Tower located?",
            "es": "¿Dónde se encuentra la torre Random House?",
            "pt": "Onde fica a torre Random House?",
        },
        {
            "id": "esma_sultan_city",
            "en": "In which city is the Esma Sultan Mansion located?",
            "es": "¿En qué ciudad se encuentra la mansión Esma Sultan?",
            "pt": "Em que cidade fica a mansão Esma Sultan?",
        },
        {
            "id": "capital_portugal",
            "en": "What is the capital of Portugal?",
            "es": "¿Cuál es la capital de Portugal?",
            "pt": "Qual é a capital de Portugal?",
        },
    ]


def load_queries(path: str | None) -> List[Dict[str, str]]:
    if not path:
        return default_queries()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Queries file must be a JSON list of {id,en,es,pt}")
        out = []
        for item in data:
            if not all(k in item for k in ("id", "en", "es", "pt")):
                continue
            out.append({"id": str(item["id"]), "en": str(item["en"]), "es": str(item["es"]), "pt": str(item["pt"])})
        return out or default_queries()


def plot_overlaps(per_query_triplets: List[Tuple[str, float, float, float]], out_png: str) -> None:
    labels = [qid for (qid, _, _, _) in per_query_triplets]
    en_es = [x for (_, x, _, _) in per_query_triplets]
    en_pt = [x for (_, _, x, _) in per_query_triplets]
    es_pt = [x for (_, _, _, x) in per_query_triplets]

    x = list(range(len(labels)))
    w = 0.25
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.bar([i - w for i in x], en_es, width=w, label="EN–ES")
    ax.bar(x, en_pt, width=w, label="EN–PT")
    ax.bar([i + w for i in x], es_pt, width=w, label="ES–PT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Jaccard overlap of top-k IDs")
    ax.set_title("Multilingual retrieval overlap (EN/ES/PT)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Multilingual retrieval smoke test (EN/ES/PT)")
    ap.add_argument("--file", type=str, default="", help="Optional JSON file with list of {id,en,es,pt}")
    ap.add_argument("--k", type=int, default=5, help="Top-k contexts per query")
    ap.add_argument("--limit", type=int, default=3, help="How many queries from the set to run")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="runtime/evals_multi/retrieval",
        help="Output directory for plots/CSV",
    )
    ap.add_argument("--save-csv", action="store_true", help="Also write multilingual_overlap.csv")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    queries = load_queries(args.file)[: max(1, args.limit)]

    per_query_triplets: List[Tuple[str, float, float, float]] = []

    for q in queries:
        qid = q["id"]
        en_q, es_q, pt_q = q["en"], q["es"], q["pt"]

        # Retrieve with local retriever (search_multi returns hits with unique chunk ids)
        en_hits = search_multi(en_q, k=args.k, k_per_entity=3, max_pool=24)
        es_hits = search_multi(es_q, k=args.k, k_per_entity=3, max_pool=24)
        pt_hits = search_multi(pt_q, k=args.k, k_per_entity=3, max_pool=24)

        en_ids = [h.id for h in en_hits]
        es_ids = [h.id for h in es_hits]
        pt_ids = [h.id for h in pt_hits]

        j_en_es = jaccard(en_ids, es_ids)
        j_en_pt = jaccard(en_ids, pt_ids)
        j_es_pt = jaccard(es_ids, pt_ids)

        per_query_triplets.append((qid, j_en_es, j_en_pt, j_es_pt))
        print(f"{qid}: EN–ES={j_en_es:.2f}  EN–PT={j_en_pt:.2f}  ES–PT={j_es_pt:.2f}")

    if per_query_triplets:
        avg_en_es = sum(x for (_, x, _, _) in per_query_triplets) / len(per_query_triplets)
        avg_en_pt = sum(x for (_, _, x, _) in per_query_triplets) / len(per_query_triplets)
        avg_es_pt = sum(x for (_, _, _, x) in per_query_triplets) / len(per_query_triplets)
        print(f"Averages: EN–ES={avg_en_es:.2f}  EN–PT={avg_en_pt:.2f}  ES–PT={avg_es_pt:.2f}")

    out_png = os.path.join(args.out_dir, "multilingual_overlap.png")
    plot_overlaps(per_query_triplets, out_png)
    print(f"[OK] Wrote {out_png}")

    if args.save_csv:
        out_csv = os.path.join(args.out_dir, "multilingual_overlap.csv")
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("id,en_es,en_pt,es_pt\n")
            for qid, a, b, c in per_query_triplets:
                f.write(f"{qid},{a:.6f},{b:.6f},{c:.6f}\n")
        print(f"[OK] Wrote {out_csv}")


if __name__ == "__main__":
    main()
