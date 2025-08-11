from datasets import load_dataset
import os, json, pathlib

OUT_DIR = pathlib.Path(__file__).parent.resolve().parent / "runtime" / "data" / "raw" / "hotpot"
os.makedirs(OUT_DIR, exist_ok=True)

# Allow remote code to run
ds = load_dataset("hotpot_qa", "distractor", split="validation[:1%]", trust_remote_code=True)

out_path = os.path.join(OUT_DIR, "hotpot_validation_1pct.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(json.dumps({k: ex[k] for k in ["question", "answer", "context"]}, ensure_ascii=False) + "\n")

print("Wrote:", out_path, "rows:", len(ds))
