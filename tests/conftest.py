import os
import sys
from pathlib import Path


def pytest_sessionstart(session):
    # Ensure project root and the scripts/ folder are importable for tests
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(scripts_dir))

    # Provide safe defaults to avoid networked behavior in tests
    os.environ.setdefault("RETRIEVER_BACKEND", "pgvector")
    os.environ.setdefault("EMBEDDING_BACKEND", "local")
    os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-m3")
