#!/usr/bin/env python3
"""
04b_upsert_chunks_pinecone.py

Use Pinecone **integrated inference** so Pinecone embeds your text for you.
- Creates an integrated index if missing (or converts an existing one with --configure-existing).
- Reads chunks from a JSONL file.
- Upserts text records in batches (<= 96) via `index.upsert_records(...)`.

Usage:
  python scripts/04b_upsert_chunks_pinecone.py \
      --index-name rag-integrated \
      --namespace __default__ \
      --chunks artifacts/chunks.jsonl \
      --model multilingual-e5-large \
      --text-field chunk_text \
      --cloud aws --region us-east-1 \
      --configure-existing

Requirements:
  pip install -U pinecone
  export PINECONE_API_KEY=...
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Iterable, List
import dotenv

dotenv.load_dotenv()

# Pinecone SDK v6+ (integrated inference)
from pinecone import Pinecone

# Optional typed helpers (SDK accepts plain strings too; import may fail on older SDKs)
try:
    from pinecone import IndexEmbed  # create_index_for_model embed config
except Exception:  # pragma: no cover
    IndexEmbed = None  # we'll pass a plain dict instead if not available


# ----------------------------
# Utilities
# ----------------------------

def sanitize_value(v):
    # Allowed: str, int/float, bool, list[str]
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        # Pinecone allows list[str] only. Coerce everything to str.
        return [str(x) for x in v]
    # dicts or other types → store as JSON string
    return json.dumps(v, ensure_ascii=False)

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def to_record(
    chunk: Dict,
    text_field_out: str = "chunk_text",
    id_keys: List[str] = None,
    text_keys: List[str] = None,
) -> Dict:
    id_keys = id_keys or ["_id", "id", "chunk_id", "doc_id"]
    text_keys = text_keys or ["chunk_text", "text", "content", "chunk", "passage", "body"]

    # ID
    rid = None
    for k in id_keys:
        if k in chunk and chunk[k] not in (None, ""):
            rid = str(chunk[k])
            break

    # Text
    text_val = None
    picked_text_key = None
    for k in text_keys:
        if k in chunk and chunk[k] not in (None, ""):
            text_val = str(chunk[k])
            picked_text_key = k
            break
    if text_val is None:
        raise ValueError(f"Chunk missing text field (tried {text_keys}). Keys: {list(chunk.keys())}")

    # Build record
    rec = {"_id": rid or f"auto-{abs(hash(text_val))}", text_field_out: text_val}

    # Keys to skip entirely (big or irrelevant for metadata)
    SKIP_KEYS = {
        picked_text_key, "embedding", "values", "vector", "dense", "sparse",
        "bm25", "score", "scores", "similarity", "distance"
    }

    for k, v in chunk.items():
        if k in id_keys or k in SKIP_KEYS:
            continue
        sv = sanitize_value(v)
        if sv is not None:
            rec[k] = sv
        # If sv is None, drop it (this fixes tokens=null and similar)

    return rec


def batch_iter(it: Iterable[Dict], size: int) -> Iterable[List[Dict]]:
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_integrated_index(
    pc: Pinecone,
    index_name: str,
    cloud: str,
    region: str,
    model: str,
    text_field: str,
    metric: str = "cosine",
    configure_existing: bool = False,
) -> str:
    """
    Returns the index host. Creates an integrated index if missing.
    If the index exists without embed config and `configure_existing` is True,
    converts it to an integrated index.
    """
    # Helper to create the embed config as either typed object or dict
    def mk_embed(model: str, text_field: str, metric: str):
        if IndexEmbed is not None:
            return IndexEmbed(model=model, field_map={"text": text_field}, metric=metric)
        return {"model": model, "field_map": {"text": text_field}, "metric": metric}

    # Create if absent
    try:
        has_index = pc.has_index(index_name)
    except Exception:
        # Fallback for older SDKs: list and check
        has_index = any(ix.name == index_name for ix in pc.list_indexes())

    if not has_index:
        eprint(f"[create] Creating integrated index '{index_name}' (cloud={cloud}, region={region}, model={model}) ...")
        desc = pc.create_index_for_model(
            name=index_name,
            cloud=cloud,         # e.g. "aws" or "gcp"
            region=region,       # e.g. "us-east-1"
            embed=mk_embed(model, text_field, metric),
        )
        return getattr(desc, "host", desc.get("host") if isinstance(desc, dict) else None)

    # If exists, verify it's integrated
    desc = pc.describe_index(name=index_name)
    host = getattr(desc, "host", desc.get("host") if isinstance(desc, dict) else None)
    embed_cfg = getattr(desc, "embed", desc.get("embed") if isinstance(desc, dict) else None)

    if embed_cfg is None:
        if not configure_existing:
            raise RuntimeError(
                f"Index '{index_name}' exists but is NOT configured for integrated embedding.\n"
                f"Run again with --configure-existing to convert it, or create a new index name."
            )
        eprint(f"[configure] Converting existing index '{index_name}' to integrated (model={model}) ...")
        pc.configure_index(
            name=index_name,
            embed=mk_embed(model, text_field, metric),
        )
        # Wait a moment for config to apply (lightweight polling)
        for _ in range(40):
            time.sleep(1.0)
            d2 = pc.describe_index(name=index_name)
            if getattr(d2, "embed", d2.get("embed") if isinstance(d2, dict) else None):
                break
        eprint("[configure] Done.")
    else:
        # Optional sanity check: warn if field_map mismatch
        fm = embed_cfg.get("field_map") if isinstance(embed_cfg, dict) else getattr(embed_cfg, "field_map", {})
        if fm and fm.get("text") != text_field:
            eprint(f"[warn] This index maps text -> '{fm.get('text')}', but you passed --text-field '{text_field}'. "
                   f"Upserts will read from '{fm.get('text')}'. Either adjust --text-field or reconfigure the index.")

    return host


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Upsert text chunks to Pinecone using integrated embedding.")

    ap.add_argument("--index-name", default=os.getenv("PINECONE_INDEX", "rag-integrated"))
    ap.add_argument("--namespace",  default=os.getenv("PINECONE_NAMESPACE", "__default__"))
    ap.add_argument("--chunks",     required=True, help="Path to JSONL with chunk dicts")

    ap.add_argument("--model",      default=os.getenv("EMBEDDING_MODEL", "multilingual-e5-large"))
    ap.add_argument("--text-field", default=os.getenv("TEXT_FIELD", "chunk_text"))

    ap.add_argument("--metric",     default=os.getenv("PINECONE_METRIC", "cosine"), choices=["cosine", "dotproduct"])
    ap.add_argument("--cloud",      default=os.getenv("PINECONE_CLOUD", "aws"))
    ap.add_argument("--region",     default=os.getenv("PINECONE_REGION", "us-east-1"))

    ap.add_argument("--batch-size", type=int, default=int(os.getenv("PINECONE_MAX_BATCH", "96")))
    ap.add_argument("--configure-existing", action="store_true",
                    help="If the index exists without embed config, convert it to integrated")
    args = ap.parse_args()


    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        eprint("ERROR: PINECONE_API_KEY is not set.")
        sys.exit(2)

    if args.batch_size > 96:
        eprint("[warn] Reducing --batch-size to 96 (text upsert limit).")
        args.batch_size = 96

    pc = Pinecone(api_key=api_key)

    # Ensure we have an integrated index, get its host, and open the data client
    host = ensure_integrated_index(
        pc=pc,
        index_name=args.index_name,
        cloud=args.cloud,
        region=args.region,
        model=args.model,
        text_field=args.text_field,
        metric=args.metric,
        configure_existing=args.configure_existing,
    )
    if not host:
        raise RuntimeError("Could not resolve index host.")

    index = pc.Index(host=host)

    # Stream chunks -> records
    src = args.chunks
    if not os.path.exists(src):
        raise FileNotFoundError(f"No such chunks file: {src}")

    eprint(f"[read] Loading chunks from: {src}")
    total, sent = 0, 0

    def record_stream():
        for chunk in read_jsonl(src):
            yield to_record(chunk, text_field_out=args.text_field)

    # Upsert in batches
    for i, batch in enumerate(batch_iter(record_stream(), args.batch_size), start=1):
        # Basic retry/backoff loop
        for attempt in range(5):
            try:
                index.upsert_records(namespace=args.namespace, records=batch)
                break
            except Exception as e:
                msg = str(e)
                # Common misconfig error
                if "Integrated inference is not configured" in msg:
                    eprint("\nERROR: Index is not integrated. Re-run with --configure-existing or use a new --index-name.")
                    raise
                # Too large / 400 / 413 type errors -> shrink batch
                if "Batch size exceeds" in msg or "Payload Too Large" in msg or "2 MB" in msg or "413" in msg:
                    if len(batch) > 1:
                        half = max(1, len(batch) // 2)
                        eprint(f"[split] Shrinking batch from {len(batch)} -> {half} due to size limits.")
                        # resend smaller half and keep remainder for next outer loop
                        small = batch[:half]
                        rest = batch[half:]
                        index.upsert_records(namespace=args.namespace, records=small)
                        # requeue the rest by processing immediately
                        index.upsert_records(namespace=args.namespace, records=rest)
                        break
                # For 429s / transient
                if "429" in msg or "rate limit" in msg or "temporarily" in msg:
                    sleep_s = 2 ** attempt
                    eprint(f"[retry] {msg} (attempt {attempt+1}/5). Sleeping {sleep_s}s ...")
                    time.sleep(sleep_s)
                    continue
                # Unknown: raise
                raise
        total += len(batch)
        sent += 1
        if sent % 10 == 0:
            eprint(f"[upsert] Sent {sent} batches / {total} records ...")

    eprint(f"[done] Upserted {total} records to index='{args.index_name}', namespace='{args.namespace}'.")


if __name__ == "__main__":
    main()
