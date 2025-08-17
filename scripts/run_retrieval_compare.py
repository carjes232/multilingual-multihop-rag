#!/usr/bin/env python3
"""
Run retrieval comparisons for pgvector (local) vs Pinecone across EN/ES/PT.

This orchestrates:
  - Optional build of a multilingual set (EN/ES/PT)
  - Recall@k evals for EN (Hotpot slice) and ES/PT (multilingual set)
  - Multilingual Jaccard overlap smoke test per backend

Outputs live under runtime/evals_multi/retrieval/ in per-backend folders.

Usage examples:
  python scripts/run_retrieval_compare.py \
    --file-en runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
    --out-root runtime/evals_multi/retrieval

  # Generate multilingual set if missing, then split to _en/_es/_pt
  python scripts/run_retrieval_compare.py --build-multilingual

Env expectations:
  - Local backend: Postgres + pgvector populated by the pipeline
  - Pinecone backend: PINECONE_* set and index populated
  - Embeddings: use multilingual-e5-large for apples-to-apples
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()  # <-- make .env available to this script and its subprocesses


def run(cmd: List[str], env_overrides: Dict[str, str] | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout if len(res.stdout) < 4000 else res.stdout[-4000:])
    if res.returncode != 0 and res.stderr:
        print("[stderr]" if len(res.stderr) < 4000 else "[stderr tail]", file=sys.stderr)
        print(res.stderr if len(res.stderr) < 4000 else res.stderr[-4000:], file=sys.stderr)
    return res.returncode, res.stdout or "", res.stderr or ""


def ensure_multilingual_set(in_file: str, out_file: str, model: str) -> None:
    if os.path.exists(out_file):
        return
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print(f"[info] Building multilingual set: {out_file}")
    code, _, _ = run(
        [
            sys.executable,
            "scripts/make_multilingual_eval_set.py",
            "--in-file",
            in_file,
            "--out-file",
            out_file,
            "--n",
            "30",
            "--seed",
            "42",
            "--model",
            model,
        ]
    )
    if code != 0:
        raise SystemExit("Failed to build multilingual set (OpenRouter config required)")


def split_multilingual_by_lang(all_path: str) -> Dict[str, str]:
    base, ext = os.path.splitext(all_path)
    outs = {
        "en": f"{base}_en{ext}",
        "es": f"{base}_es{ext}",
        "pt": f"{base}_pt{ext}",
    }
    if all(os.path.exists(p) for p in outs.values()):
        return outs
    print(f"[info] Splitting multilingual set by language: {all_path}")
    rows: Dict[str, List[dict]] = {"en": [], "es": [], "pt": []}
    with open(all_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            lang = (obj.get("lang") or "en").strip().lower()
            if lang not in rows:
                continue
            rows[lang].append(obj)
    for lang, path in outs.items():
        with open(path, "w", encoding="utf-8") as f:
            for o in rows[lang]:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"[ok] Wrote {path} ({len(rows[lang])} rows)")
    return outs


def eval_retrieval_for(
    env_overrides: Dict[str, str], file_path: str, out_dir: str, k_list: str = "1,5,10", limit: int = 200
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    code, _, _ = run(
        [
            sys.executable,
            "scripts/eval_retrieval.py",
            "--file",
            file_path,
            "--k-list",
            k_list,
            "--limit",
            str(limit),
            "--plot-out",
            out_dir,
            "--save-csv",
        ],
        env_overrides,
    )
    if code != 0:
        raise SystemExit(f"eval_retrieval failed for {file_path} -> {out_dir}")


def multilingual_overlap_for(env_overrides: Dict[str, str], out_dir: str, k: int = 5, limit: int = 3) -> None:
    os.makedirs(out_dir, exist_ok=True)
    code, _, _ = run(
        [
            sys.executable,
            "scripts/eval_multilingual_retrieval.py",
            "--k",
            str(k),
            "--limit",
            str(limit),
            "--out-dir",
            out_dir,
            "--save-csv",
        ],
        env_overrides,
    )
    if code != 0:
        raise SystemExit(f"multilingual overlap failed for {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Compare local vs Pinecone retrieval (EN/ES/PT)")
    ap.add_argument(
        "--file-en",
        default="runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl",
        help="English eval JSONL",
    )
    ap.add_argument(
        "--file-multilingual",
        default="runtime/data/raw/hotpot/hotpot_multilingual_10.jsonl",
        help=("Multilingual (EN/ES/PT) JSONL; will be created with --build-multilingual if missing"),
    )
    ap.add_argument(
        "--build-multilingual",
        action="store_true",
        help="Create multilingual set if missing (requires OpenRouter)",
    )
    ap.add_argument(
        "--openrouter-model",
        default="google/gemini-2.0-flash-001",
        help="Model for translation when building set",
    )
    ap.add_argument(
        "--out-root",
        default="runtime/evals_multi/retrieval",
        help="Output root directory for plots/CSVs",
    )
    ap.add_argument("--skip-pinecone", action="store_true", help="Skip Pinecone runs (if not configured)")
    ap.add_argument("--skip-local", action="store_true", help="Skip local (pgvector) runs (if DB not available)")
    ap.add_argument("--k-list", default="1,5,10", help="k list for recall evals")
    ap.add_argument("--limit-en", type=int, default=200, help="Limit for EN Hotpot slice eval")
    ap.add_argument("--limit-multi", type=int, default=0, help="Limit for ES/PT multilingual eval (0=all)")
    args = ap.parse_args()

    # Optionally create multilingual set
    if args.build_multilingual or not os.path.exists(args.file_multilingual):
        ensure_multilingual_set(args.file_en, args.file_multilingual, args.openrouter_model)

    # Split multilingual by language
    outs = split_multilingual_by_lang(args.file_multilingual)
    file_en_multi = outs["en"]
    file_es = outs["es"]
    file_pt = outs["pt"]

    # Environments for each backend
    env_local = {
        "RETRIEVER_BACKEND": "pgvector",
        "EMBEDDING_BACKEND": "local",
        "EMBEDDING_MODEL": "multilingual-e5-large",
    }
    env_pc = {
        "RETRIEVER_BACKEND": "pinecone",
        "EMBEDDING_BACKEND": "pinecone",
        "EMBEDDING_MODEL": "multilingual-e5-large",
        # Expect PINECONE_* already set in env/.env (loaded above)
    }

    # Run Local
    if not args.skip_local:
        eval_retrieval_for(env_local, args.file_en, os.path.join(args.out_root, "local_en"), args.k_list, args.limit_en)
        eval_retrieval_for(
            env_local, file_en_multi, os.path.join(args.out_root, "local_en"), args.k_list, args.limit_multi
        )
        eval_retrieval_for(env_local, file_es, os.path.join(args.out_root, "local_es"), args.k_list, args.limit_multi)
        eval_retrieval_for(env_local, file_pt, os.path.join(args.out_root, "local_pt"), args.k_list, args.limit_multi)
        multilingual_overlap_for(env_local, os.path.join(args.out_root, "local"))
    else:
        print("[skip] Local (pgvector) runs disabled via --skip-local")

    # Run Pinecone
    if not args.skip_pinecone:
        pc_idx = os.getenv("PINECONE_INDEX", "<unset>")
        pc_ns = os.getenv("PINECONE_NAMESPACE", "__default__") or "__default__"
        print(f"[info] Pinecone index={pc_idx} namespace={pc_ns}")
        eval_retrieval_for(env_pc, args.file_en, os.path.join(args.out_root, "pinecone_en"), args.k_list, args.limit_en)
        eval_retrieval_for(
            env_pc, file_en_multi, os.path.join(args.out_root, "pinecone_en"), args.k_list, args.limit_multi
        )
        eval_retrieval_for(env_pc, file_es, os.path.join(args.out_root, "pinecone_es"), args.k_list, args.limit_multi)
        eval_retrieval_for(env_pc, file_pt, os.path.join(args.out_root, "pinecone_pt"), args.k_list, args.limit_multi)
        multilingual_overlap_for(env_pc, os.path.join(args.out_root, "pinecone"))
    else:
        print("[skip] Pinecone runs disabled via --skip-pinecone")

    print("[DONE] Outputs under:", os.path.abspath(args.out_root))


if __name__ == "__main__":
    main()
