PY=python
PIP=pip

# Default env
VENV=.venv
ACT=. $(VENV)/bin/activate

.PHONY: help install venv api pipeline index eval-local eval-online search db-up db-down format lint test pinecone-upsert pinecone-reindex pinecone-health pinecone-recreate

help:
	@echo "Targets:"
	@echo "  venv          - create venv (.venv)"
	@echo "  install       - install python deps (requirements.txt)"
	@echo "  db-up         - start Postgres (docker compose)"
	@echo "  db-down       - stop Postgres"
	@echo "  pipeline      - run 01..05 (download, chunk, load, index, embed)"
	@echo "  index         - create pgvector HNSW index"
	@echo "  api           - run FastAPI app on :8000"
	@echo "  search        - example CLI search"
	@echo "  eval-local    - local RAG vs No-RAG eval (Ollama)"
	@echo "  eval-online   - online eval (OpenRouter)"
	@echo "  pinecone-upsert - upsert chunks to Pinecone (online embeddings)"
	@echo "  pinecone-reindex - create Pinecone index with source_model + upsert"
	@echo "  pinecone-health  - run Pinecone health + smoke search"
	@echo "  pinecone-recreate - delete and recreate Pinecone index (DANGEROUS)"
	@echo "  lint          - ruff check"
	@echo "  format        - ruff format"
	@echo "  test          - pytest -q"

venv:
	@test -d $(VENV) || $(PY) -m venv $(VENV)

install: venv
	$(ACT) && $(PIP) install -r requirements.txt

db-up:
	docker compose up -d db

db-down:
	docker compose down

pipeline:
	$(ACT) && $(PY) scripts/01_download_hotpot.py
	$(ACT) && $(PY) scripts/02_chunk_hotpot.py
	$(ACT) && $(PY) scripts/03_load_docs_chunks.py
	$(ACT) && psql -h localhost -p 5432 -U rag -d ragdb -f scripts/05_create_vector_index.sql
	$(ACT) && $(PY) scripts/04_embed_chunks.py

index:
	$(ACT) && psql -h localhost -p 5432 -U rag -d ragdb -f scripts/05_create_vector_index.sql

api:
	$(ACT) && uvicorn scripts.api:app --reload --port 8000

search:
	$(ACT) && $(PY) scripts/search.py "Where is the Random House Tower located?" --k 5

eval-local:
	$(ACT) && $(PY) scripts/eval_models.py --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
	  --out-dir runtime/evals_multi --sample 50 --k 4 --models "qwen3:4b,gemma3:1b,gemma3:4b,deepseek-r1:8b" --write-errors

eval-online:
	$(ACT) && $(PY) scripts/eval_models_online.py --file runtime/data/raw/hotpot/hotpot_validation_1pct.jsonl \
	  --out-dir runtime/evals_multi --sample 50 --k 4 \
	  --models "google/gemini-2.0-flash-001,google/gemini-2.5-flash" --write-errors

lint:
	$(ACT) && ruff check .

format:
	$(ACT) && ruff format .

test:
	$(ACT) && pytest -q

pinecone-upsert:
	$(ACT) && $(PY) scripts/04b_upsert_chunks_pinecone.py --index $${PINECONE_INDEX:-rag} --batch 128 --limit 0

pinecone-reindex:
	$(ACT) && $(PY) scripts/04b_upsert_chunks_pinecone.py --create-index --index $${PINECONE_INDEX:-rag} --batch 128 --limit 0

pinecone-health:
	$(ACT) && $(PY) scripts/pinecone_health.py --index $${PINECONE_INDEX:-rag}

# Danger: deletes and recreates the index when it already exists
pinecone-recreate:
	$(ACT) && $(PY) scripts/04b_upsert_chunks_pinecone.py --create-index --force-recreate --index $${PINECONE_INDEX:-rag} --batch 128 --limit 0
