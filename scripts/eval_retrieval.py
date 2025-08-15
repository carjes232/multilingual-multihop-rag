#!/usr/bin/env python3
"""
Quick retrieval eval with optional plots.

Computes recall@k using Hotpot titles as gold labels and can
produce diagnostic plots to visualize retrieval efficiency.

Assumes you have loaded/embedded chunks and created the HNSW index.

Examples:
  # Basic evaluation
  python scripts/eval_retrieval.py --k 5 --limit 50 \
    --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl

  # Evaluate multiple k values and save plots/CSV under runtime/evals_multi/retrieval
  python scripts/eval_retrieval.py --k-list 1,5,10 --limit 200 --plot-out runtime/evals_multi/retrieval \
    --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl

Notes:
  - Plotting uses matplotlib if available. If not installed and plotting is requested,
    the script will print an instruction to install it: pip install matplotlib
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import embedder as emb
import retriever as retr
import sys

# Make stdout/stderr tolerant to non-ASCII on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def parse_k_list(k_list_str: str, fallback_k: int) -> List[int]:
    """Parse comma-separated k list, or fall back to single k."""
    if not k_list_str:
        return [fallback_k]
    vals: List[int] = []
    for part in k_list_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid k value in --k-list: {part!r}")
    vals = sorted({v for v in vals if v > 0})
    return vals or [fallback_k]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_and_save(
    out_dir: str,
    k_values: List[int],
    recall_by_k: Dict[int, float],
    ranks: List[Optional[int]],
    avg_times_ms: Dict[str, float],
    save_csv: bool,
) -> None:
    """Create plots and optional CSV/summary under out_dir.

    Lazy-imports matplotlib so non-plotting runs don't require it.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("Plotting requested but matplotlib not installed. Run: pip install matplotlib")
        return

    ensure_dir(out_dir)

    # 1) Recall vs k
    ks = sorted(k_values)
    recalls = [recall_by_k.get(k, 0.0) for k in ks]
    plt.figure(figsize=(5, 3.2), dpi=150)
    plt.plot(ks, recalls, marker="o")
    for x, y in zip(ks, recalls):
        plt.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    plt.title("Recall@k")
    plt.xlabel("k")
    plt.ylabel("Recall")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recall_vs_k.png"), bbox_inches="tight")
    plt.close()

    # 2) Histogram of first-hit rank
    ranks_hit = [r for r in ranks if r and r > 0]
    if ranks_hit:
        k_max = max(ks)
        bins = range(1, k_max + 2)
        plt.figure(figsize=(5, 3.2), dpi=150)
        plt.hist(ranks_hit, bins=bins, align="left", rwidth=0.85)
        plt.title("First-Hit Rank Distribution")
        plt.xlabel("Rank of first hit")
        plt.ylabel("Count")
        plt.xticks(list(range(1, k_max + 1)))
        plt.grid(True, alpha=0.3, axis="y", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hit_rank_hist.png"), bbox_inches="tight")
        plt.close()

    # 3) Average timings
    labels = ["encode_ms", "db_ms", "total_ms"]
    vals = [avg_times_ms.get(k, 0.0) for k in labels]
    plt.figure(figsize=(5, 3.2), dpi=150)
    bars = plt.bar(labels, vals, color=["#4e79a7", "#f28e2b", "#76b7b2"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.title("Average Latency per Query (ms)")
    plt.ylabel("Milliseconds")
    plt.grid(True, alpha=0.3, axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avg_times_ms.png"), bbox_inches="tight")
    plt.close()

    # Optional CSV and summary JSON
    if save_csv:
        csv_path = os.path.join(out_dir, "recall_by_k.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("k,recall\n")
            for k, r in zip(ks, recalls):
                f.write(f"{k},{r:.6f}\n")

    summary = {
        "recall_by_k": {int(k): float(v) for k, v in recall_by_k.items()},
        "avg_times_ms": {k: float(v) for k, v in avg_times_ms.items()},
        "num_hits": int(len([r for r in ranks if r and r > 0])),
        "num_eval": int(len(ranks)),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def vec_to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def maybe_load_model():
    # Use embedder's loader for consistent model id and E5 handling
    if emb.get_backend() == "local":
        try:
            return emb._ensure_local_model()  # type: ignore[attr-defined]
        except Exception:
            return None
    return None



def gold_titles_from_example(ex: dict) -> Set[str]:
    ctx = ex.get("context", {})
    titles = ctx.get("title", [])
    if isinstance(titles, list):
        return {str(t).strip() for t in titles if t}
    return set()


def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval recall@k using HotpotQA titles")
    ap.add_argument("--file", default="runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl",
                    help="Hotpot JSONL with fields question/answer/context")
    ap.add_argument("--k", dest="top_k", type=int, default=5, help="Top-K if --k-list not provided")
    ap.add_argument("--k-list", type=str, default="", help="Comma-separated list of K values (e.g., '1,5,10'). Overrides --k.")
    ap.add_argument("--limit", type=int, default=50, help="How many questions to evaluate")
    ap.add_argument("--plot-out", type=str, default=None, help="Directory to save plots/CSV (optional)")
    ap.add_argument("--save-csv", action="store_true", help="Also write recall_by_k.csv when plotting")
    args = ap.parse_args()

    model = maybe_load_model()
    r = retr.make_retriever()

    k_values = parse_k_list(args.k_list, args.top_k)
    k_max = max(k_values)

    total = 0
    hits_by_k: Dict[int, int] = {k: 0 for k in k_values}
    mrr_sum = 0.0
    ranks: List[Optional[int]] = []
    examples = []

    encode_ms_total = 0.0
    db_ms_total = 0.0
    total_ms_total = 0.0

    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            if args.limit and total >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            q = ex.get("question")
            gold_titles = gold_titles_from_example(ex)
            if not q or not gold_titles:
                continue

            t0_total = time.perf_counter()

            if emb.get_backend() == "local":
                t0_enc = time.perf_counter()
                # Embed via embedder to ensure E5 'query:' prefix and correct model id
                qvec = emb.embed([q], purpose="query")[0]
                t1_enc = time.perf_counter()
                t0_db = time.perf_counter()
                hits_batches = r.search_vecs(np.expand_dims(qvec, 0), k_max)  # type: ignore[attr-defined]
                hits = hits_batches[0] if hits_batches else []
                t1_db = time.perf_counter()
            else:
                # Pinecone integrated search: no client-side embedding
                t0_enc = t1_enc = time.perf_counter()
                t0_db = time.perf_counter()
                hits_batches = r.search(None, [q], k_max)
                hits = hits_batches[0] if hits_batches else []
                t1_db = time.perf_counter()

            ret_titles_list = [h.title for h in hits if h.title]

            # hits by k
            for k in k_values:
                if gold_titles & set(ret_titles_list[:k]):
                    hits_by_k[k] += 1

            # first-hit rank and MRR
            rank: Optional[int] = None
            for i, t in enumerate(ret_titles_list, start=1):
                if t in gold_titles:
                    rank = i
                    break
            ranks.append(rank)
            if rank:
                mrr_sum += 1.0 / rank

            t1_total = time.perf_counter()
            encode_ms_total += (t1_enc - t0_enc) * 1000.0
            db_ms_total += (t1_db - t0_db) * 1000.0
            total_ms_total += (t1_total - t0_total) * 1000.0

            total += 1
            if total <= 5:  # keep a few samples for display
                examples.append((q, list(gold_titles)[:3], [(h.title, float(h.score)) for h in hits[:3]], rank is not None))

    recall_by_k = {k: (hits_by_k[k] / total) if total else 0.0 for k in k_values}
    mrr = (mrr_sum / total) if total else 0.0
    avg_times_ms = {
        "encode_ms": (encode_ms_total / total) if total else 0.0,
        "db_ms": (db_ms_total / total) if total else 0.0,
        "total_ms": (total_ms_total / total) if total else 0.0,
    }

    print(f"Evaluated: {total} questions | k-values={k_values}")
    recall_summary = ", ".join([f"R@{k}={recall_by_k[k]:.3f}" for k in sorted(k_values)])
    print(f"{recall_summary} | MRR@{k_max}={mrr:.3f}  (hits={hits_by_k})")
    print(f"Avg times (ms): encode={avg_times_ms['encode_ms']:.1f}, db={avg_times_ms['db_ms']:.1f}, total={avg_times_ms['total_ms']:.1f}\n")

    print("Examples:")
    for i, (q, gold, retrieved, ok) in enumerate(examples, 1):
        print(f"{i}. {'HIT ' if ok else 'MISS'} q={q}")
        print(f"   gold_titles: {gold}")
        print(f"   top3: {retrieved}\n")

    if args.plot_out:
        plot_and_save(
            out_dir=args.plot_out,
            k_values=k_values,
            recall_by_k=recall_by_k,
            ranks=ranks,
            avg_times_ms=avg_times_ms,
            save_csv=args.save_csv,
        )

    # no connection to close for env-gated retriever


if __name__ == "__main__":
    main()
