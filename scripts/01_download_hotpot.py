import json
import os
import pathlib

from datasets import load_dataset

OUT_DIR = pathlib.Path(__file__).parent.resolve().parent / "runtime" / "data" / "raw" / "hotpot"
os.makedirs(OUT_DIR, exist_ok=True)

# Safer default: do not trust remote code unless explicitly allowed
ALLOW_REMOTE_CODE = (os.getenv("ALLOW_REMOTE_CODE") or "").strip() in {"1", "true", "yes"}
ds = load_dataset(
    "hotpot_qa",
    "distractor",
    split="validation[:1%]",
    trust_remote_code=ALLOW_REMOTE_CODE,
)

out_path = os.path.join(OUT_DIR, "hotpot_validation_1pct.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(json.dumps({k: ex[k] for k in ["question", "answer", "context"]}, ensure_ascii=False) + "\n")

print("Wrote:", out_path, "rows:", len(ds))
