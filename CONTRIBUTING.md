# Contributing

Thanks for your interest! This project aims to be a clear, reproducible RAG showcase.

## Ground Rules
- Use Conventional Commits (e.g., `feat(api): ...`, `fix(search): ...`).
- Keep diffs focused; do not reformat unrelated files.
- Follow existing style and config (`ruff`, `pyproject.toml`).
- Do not commit secrets. Use `.env` (ignored) and update `.env.example` for new keys.

## Setup
```bash
make setup            # create venv and install requirements
make lint             # ruff check
make format           # ruff format
make test             # pytest -q
```

## Tests
- Add/modify tests under `tests/`.
- Mark heavy/online tests as optional (env‑gated). See `tests/optional/*` for examples.

## Code Style
- Python 3.10+, 4‑space indent, short docstrings.
- Respect the configured line length and imports order (`ruff`).

## Security
- Never commit `.env` or credentials.
- If you suspect a leak, rotate the key and open a brief issue.

