#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import uuid
from collections import defaultdict

# ---------- Config ----------
# Paths (kept compatible with your layout)
ROOT = pathlib.Path(__file__).parent.resolve().parent
data_path = ROOT / "runtime" / "data" / "raw" / "hotpot"
out_path_docs = ROOT / "runtime" / "staging" / "documents.jsonl"
out_path_chunks = ROOT / "runtime" / "staging" / "chunks.jsonl"

# Chunking params
MAX_WORDS = 220
OVERLAP = 40
LANG = "en"

# ---------- Safety checks ----------
assert 0 <= OVERLAP < MAX_WORDS, "OVERLAP must be >=0 and < MAX_WORDS"
os.makedirs(out_path_docs.parent, exist_ok=True)
if not data_path.exists():
    raise FileNotFoundError(f"Input folder not found: {data_path}")


# ---------- Helpers ----------
def iter_chunks_words(words, max_words=MAX_WORDS, overlap=OVERLAP):
    """Yield chunk texts using a sliding window on words with overlap."""
    start = 0
    n = len(words)
    while start < n:
        end = min(start + max_words, n)
        yield " ".join(words[start:end]).strip()
        if end == n:
            break
        # Advance from the actual end to guarantee exact overlap
        start = end - overlap


# ---------- State ----------
title2id = {}  # title -> doc_id
next_ord = defaultdict(int)  # doc_id -> next chunk ordinal
seen_paragraphs = set()  # hashes of normalized paragraphs

docs_written = 0
chunks_written = 0
lines_seen = 0
lines_bad = 0

# ---------- Processing ----------
with open(out_path_docs, "w", encoding="utf-8") as f_docs, open(out_path_chunks, "w", encoding="utf-8") as f_chunks:
    # deterministic order helps reproducibility
    for path in sorted(data_path.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f_in:
            for raw in f_in:
                line = raw.strip()
                if not line:
                    continue
                lines_seen += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    lines_bad += 1
                    continue

                ctx = entry.get("context", {})
                titles = ctx.get("title", [])
                sents_lists = ctx.get("sentences", [])
                # Guard against shape mismatches
                if not isinstance(titles, list) or not isinstance(sents_lists, list):
                    lines_bad += 1
                    continue

                documents_len = min(len(titles), len(sents_lists))
                for i in range(documents_len):
                    title = (titles[i] or "").strip()
                    # Hotpot has list of sentences (strings) for each title index
                    sents = sents_lists[i] if isinstance(sents_lists[i], list) else []
                    paragraph_text = " ".join(s.strip() for s in sents if isinstance(s, str)).strip()
                    if not title or not paragraph_text:
                        continue

                    # Stable doc_id per unique title; write document only once
                    doc_id = title2id.get(title)
                    if doc_id is None:
                        doc_id = str(uuid.uuid4())
                        title2id[title] = doc_id
                        doc = {"id": doc_id, "title": title, "lang": LANG, "url": None, "meta": {}}
                        f_docs.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        docs_written += 1

                    # Paragraph-level dedupe across the whole run
                    para_hash = hashlib.sha1(paragraph_text.encode("utf-8")).hexdigest()
                    if para_hash in seen_paragraphs:
                        continue
                    seen_paragraphs.add(para_hash)

                    # Word-based chunking with overlap
                    words = paragraph_text.split()
                    for chunk_text in iter_chunks_words(words):
                        chunk = {
                            "id": str(uuid.uuid4()),
                            "doc_id": doc_id,
                            "ord": next_ord[doc_id],
                            "lang": LANG,
                            "title": title,
                            "text": chunk_text,
                            "tokens": None,
                        }
                        f_chunks.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        chunks_written += 1
                        next_ord[doc_id] += 1

# ---------- Summary ----------
print(
    f"Done.\n"
    f"Files processed: {len(list(data_path.glob('*.jsonl')))}\n"
    f"Lines seen: {lines_seen} | bad: {lines_bad}\n"
    f"Unique titles: {len(title2id)}\n"
    f"Documents written: {docs_written}\n"
    f"Chunks written: {chunks_written}\n"
)
