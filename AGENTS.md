# Repository Guidelines

## Project Structure & Modules
- `scripts/`: data pipeline, API, evals, tools (e.g., `api.py`, `search.py`, `eval_models.py`).
- `runtime/`: datasets, staging, logs, and eval outputs (`runtime/data/raw`, `runtime/staging`, `runtime/evals_multi`).
- Root: `requirements.txt`, `.gitignore`, optional `.env` (ignored), `.venv/`.
- Large JSONL/artefacts stay out of Git per `.gitignore`; plots/CSVs under `runtime/evals_multi/` are kept.

## Build, Run, and Dev
```bash
# Install
python -m venv .venv && . .venv/bin/activate  # or Windows equivalent
pip install -r requirements.txt

# Data pipeline
python scripts/01_download_hotpot.py
python scripts/02_chunk_hotpot.py
python scripts/03_load_docs_chunks.py
psql -h localhost -p 5432 -U rag -d ragdb -f scripts/05_create_vector_index.sql
python scripts/04_embed_chunks.py

# API (FastAPI)
uvicorn scripts.api:app --reload --port 8000
curl http://127.0.0.1:8000/health

# Quick CLI search
python scripts/search.py "Where is the Random House Tower located?" --k 5

# Retrieval evaluation (local; recall@k + plots)
python scripts/eval_retrieval.py --k-list 1,5,10 --limit 200 \
  --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
  --plot-out runtime/evals_multi/retrieval --save-csv

# RAG vs No‑RAG (local; Ollama + local /answer API)
python scripts/eval_models.py --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
  --out-dir runtime/evals_multi --sample 74 --k 4 --models "qwen3:4b,gemma3:1b,gemma3:4b,deepseek-r1:8b" --write-errors

# RAG vs No‑RAG (online; OpenRouter)
# Requires .env with OPENROUTER_API_KEY (see below)
python scripts/eval_models_online.py --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
  --out-dir runtime/evals_multi --sample 74 --k 4 \
  --models "google/gemini-2.0-flash-001,google/gemini-2.5-flash,deepseek/deepseek-chat-v3-0324:free,qwen/qwen3-30b-a3b" --write-errors

# Compare Local vs Online summaries (CSV + plots)
python scripts/compare_eval_results.py \
  --local runtime/evals_multi/model_summaries.csv \
  --online runtime/evals_multi/model_summaries_online.csv \
  --out-dir runtime/evals_multi

# Multilingual set (10×3 EN/ES/PT) from Hotpot slice
python scripts/make_multilingual_eval_set.py \
  --in-file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
  --out-file runtime/data/raw/hotpot/hotpot_multilingual_10.jsonl \
  --n 10 --seed 42 --model google/gemini-2.0-flash-001

# Multilingual retrieval overlap (EN/ES/PT)
python scripts/eval_multilingual_retrieval.py --k 5 --limit 3 \
  --out-dir runtime/evals_multi/retrieval --save-csv
```

## Coding Style & Naming
- Python 3.10+, PEP 8, 4‑space indent; `snake_case` for files/functions, `CapWords` for classes.
- Prefer type hints and short docstrings (module entry points include `main()` where applicable).
- Keep scripts self‑contained and idempotent; avoid hard‑coding secrets.
- Config: read DB/model settings from env when adding new code; never commit `.env` (use `.env.example`).

## Testing Guidelines
- No formal unit tests yet; use evaluation scripts as regression checks:
  - Retrieval: `python scripts/eval_retrieval.py --k 5 --limit 50 --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl`
  - RAG vs No‑RAG (local): `python scripts/eval_models.py ... --write-errors`
  - RAG vs No‑RAG (online): `python scripts/eval_models_online.py ... --write-errors`
  - Local vs Online comparison: `python scripts/compare_eval_results.py --local ... --online ...`
- For new modules, add a minimal smoke path (small input, deterministic seed) and document the command in the PR.

## Commit & PR Guidelines
- Use Conventional Commits (e.g., `feat(api): ...`, `fix(search): ...`).
- Scope PRs narrowly; include:
  - What/why, reproducible commands, and sample output/plots.
  - Updated README/API docs when endpoints or flags change.
  - New config in `.env.example` if applicable.
- Do not commit large JSONL, logs, or `.env`; respect `.gitignore`.

## Security & Configuration
- Database defaults live in scripts; prefer env overrides for deployments.
- Requires Postgres with `pgvector` and (recommended) CUDA GPU. Verify GPU via `python scripts/nvidia_verify.py`.
- Handle external calls (e.g., Ollama) defensively and log errors under `runtime/logs/`.

### OpenRouter configuration
- Place secrets only in `.env` (never commit):
  - `OPENROUTER_API_KEY` (required)
  - `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
  - `OPENROUTER_HTTP_REFERER`, `OPENROUTER_X_TITLE` (optional, helpful for rate/cost attribution)
- Online evaluator saves per‑model error JSONL when `--write-errors` is passed.

### Outputs and Artefacts
- Retrieval (local): `runtime/evals_multi/retrieval/recall_vs_k.png`, `hit_rank_hist.png`, `avg_times_ms.png`, optional `recall_by_k.csv`, `summary.json`.
- Local RAG vs No‑RAG: `runtime/evals_multi/model_summaries.csv`, `em.png`, `f1.png`, `latency_p50.png`, `em_breakdown.png`, `f1_gain_hist.png`, optional `errors_*.jsonl`.
- Online RAG vs No‑RAG: `runtime/evals_multi/model_summaries_online.csv`, `em_online.png`, `f1_online.png`, `latency_p50_online.png`, `em_breakdown_online.png`, `f1_gain_hist_online.png`, optional `errors_online_*.jsonl`.
- Comparison (local vs online): `combined_long.csv`, `comparison_summary.csv`, `comparison_by_model.csv` (if overlap), `*_common.png`, `env_*.png` under `runtime/evals_multi/`.
- Multilingual: `runtime/data/raw/hotpot/hotpot_multilingual_10.jsonl`, and `runtime/evals_multi/retrieval/multilingual_overlap.png` (+ CSV if requested).

## Evaluation Scripts Cheat‑Sheet
- `scripts/eval_retrieval.py`: recall@k across multiple k via `--k-list`, first‑hit rank histogram, average latency bars; optional CSV/JSON and plot directory via `--plot-out` and `--save-csv`.
- `scripts/eval_models.py`: local Ollama + local `/answer` API; computes EM, F1, p50/p95 latencies, EM breakdown, F1 gain hist; writes `model_summaries.csv` and plots.
- `scripts/eval_models_online.py`: OpenRouter versions of the above; reads `OPENROUTER_*` from `.env`; writes `model_summaries_online.csv` and plots.
- `scripts/compare_eval_results.py`: joins local and online CSVs; emits env‑level macro/weighted summaries, per‑model deltas for overlapping models, and side‑by‑side plots.
- `scripts/make_multilingual_eval_set.py`: builds a 10‑example EN/ES/PT JSONL using OpenRouter translation; keeps answers/context in English for robust EM/F1.
- `scripts/eval_multilingual_retrieval.py`: multilingual retrieval smoke test; Jaccard overlap across EN/ES/PT for top‑k IDs.

## Next Tasks (Roadmap)
- Refactor: add sane “run defaults” per script
  - Provide default flags and ergonomic entrypoints so `python scripts/<tool>.py` runs our typical configuration without long arg lists.
  - Consider a simple `make eval-local`, `make eval-online`, `make compare` to document canonical runs.
- Finish multilingual support
  - Add “answer in English” constraint to prompts for non‑EN questions for stable EM/F1.
  - Expand yes/no normalization to ES/PT (e.g., "sí/sim" ↔ yes, "no" consistent), and add locale‑aware normalization toggles.
  - Run multilingual retrieval overlap on a broader set; add per‑language metrics and plots.
- Pinecone vs Local comparison (with larger data)
  - Add Pinecone backend option to retriever (env‑gated), with `.env.example` entries for `PINECONE_API_KEY`, environment, and index.
  - Re‑embed and load a larger Hotpot slice (hundreds–thousands); compare recall@k, latency, and overall RAG vs No‑RAG performance.
  - Produce head‑to‑head plots and a comparison CSV similar to local vs online.
- Reproducible showcase runner
  - One script (or Makefile) to: generate multilingual set → run retrieval eval → run local+online RAG evals → run comparisons → drop all plots/CSVs in `runtime/evals_multi/`.
- Response caching and cost tracking (online)
  - Cache OpenRouter responses on disk for determinism and cost savings; expose a `--cache-dir` flag.
  - Optional per‑model cost summary if token usage available.
- Embedding & retrieval ablations
  - Compare BGE‑M3 vs multilingual alternatives; vary `k` and context length; report impacts on EM/F1 and latency.
- QA/error analysis artifacts
  - Aggregate and visualize typical failure modes; include top entities, frequent mismatches, and example snippets in a compact HTML report.
