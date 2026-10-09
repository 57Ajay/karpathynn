#!/usr/bin/env python3
"""Build the HuggingFace parity fixtures: a random-weight GPT-2 (tiny config),
saved as safetensors + config.json + the logits transformers computes for a
fixed input. tests/parity.rs then asserts our Rust forward matches.

Runs fully offline: the model is constructed from a local config, never
downloaded. Requires: pip install torch transformers safetensors
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2LMHeadModel

out_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/hf_tiny"
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(42)
cfg = GPT2Config(
    vocab_size=101,
    n_positions=32,
    n_embd=64,
    n_layer=2,
    n_head=2,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
)
model = GPT2LMHeadModel(cfg).eval()

# keep only real parameters: drop the causal-mask buffers and the tied lm_head copy
sd = model.state_dict()
# ".attn.bias" (dotted!) is the causal-mask buffer; "...c_attn.bias" is a real parameter
params = {
    k: v
    for k, v in sd.items()
    if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias") and k != "lm_head.weight"
}
save_file(params, f"{out_dir}/model.safetensors")

with open(f"{out_dir}/config.json", "w") as f:
    json.dump(cfg.to_dict(), f)

input_ids = [3, 14, 15, 92, 65, 35, 89, 79, 3, 23, 84, 62]
with torch.no_grad():
    logits = model(torch.tensor([input_ids])).logits  # (1, T, V)

with open(f"{out_dir}/reference.json", "w") as f:
    json.dump(
        {
            "input_ids": input_ids,
            "logits_all": [float(x) for x in logits[0].flatten().tolist()],
        },
        f,
    )
print(f"wrote fixtures to {out_dir}: {len(params)} tensors, logits {tuple(logits.shape)}")
