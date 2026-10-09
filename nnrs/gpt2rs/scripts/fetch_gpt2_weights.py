#!/usr/bin/env python3
"""Download the real GPT-2 124M weights (config.json + model.safetensors) from
HuggingFace so `gpt2 generate --hf-config ... --hf-weights ...` can run them.

    pip install huggingface_hub
    python3 scripts/fetch_gpt2_weights.py            # -> gpt2-124m/
    python3 scripts/fetch_gpt2_weights.py gpt2-medium  # 355M works too (same architecture)

Then:
    cargo run --release -- generate \
        --hf-config gpt2-124m/config.json --hf-weights gpt2-124m/model.safetensors \
        --prompt "Hello, I'm a language model," --tokens 60 --temperature 0.8 --top-k 40

Note: the official HF 'gpt2' safetensors stores f32 — ~500MB download, ~1.5GB RAM to run.
"""
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("pip install huggingface_hub first")

repo = sys.argv[1] if len(sys.argv) > 1 else "gpt2"  # the 124M model
out = "gpt2-124m" if repo == "gpt2" else repo
for fname in ["config.json", "model.safetensors"]:
    path = hf_hub_download(repo_id=repo, filename=fname, local_dir=out)
    print("fetched", path)
print(f"done. run: cargo run --release -- generate --hf-config {out}/config.json --hf-weights {out}/model.safetensors --prompt \"Hello, I'm a language model,\" --tokens 60 --top-k 40 --temperature 0.8")
