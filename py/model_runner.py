# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "tiktoken"]
# ///
"""
uv run model_runner.py --prompt "Once upon a time" --max_new_tokens 200
uv run model_runner.py                       # interactive, Ctrl-D to quit
uv run model_runner.py --weights path/to/model.pt -t 0.7 --top_p 0.9 -n 3
"""

import argparse
import os
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=-1)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_token: int | None = None,
        valid_vocab: int | None = None,
    ) -> torch.Tensor:
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[
                :, -self.config.block_size :
            ]  # wpe only has block_size positions
            logits = self(idx_cond)[:, -1, :].float()

            if (
                valid_vocab is not None
            ):  # vocab is padded to 50304; GPT-2 only has 50257
                logits[:, valid_vocab:] = float("-inf")

            if temperature <= 0:  # greedy
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    drop = (sorted_probs.cumsum(dim=-1) - sorted_probs) > top_p
                    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        1, sorted_idx, sorted_logits
                    )
                idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if stop_token is not None:
                finished |= idx_next.squeeze(1) == stop_token
                if bool(finished.all()):
                    break
        return idx


#  loading / running
def find_weights(path: str | None) -> str:
    if path:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ("model.pt", os.path.join("weights", "model.pt")):
        for base in (os.getcwd(), here):
            full = os.path.join(base, cand)
            if os.path.exists(full):
                return full
    raise FileNotFoundError("model.pt not found - pass --weights path/to/model.pt")


def load_model(path: str, device: str) -> GPT:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def generate_text(
    model: GPT, enc: tiktoken.Encoding, prompt: str, args, device: str
) -> list[str]:
    # training data had <|endoftext|> before every story
    ids = [enc.eot_token] + enc.encode_ordinary(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device).repeat(
        args.num_samples, 1
    )
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_token=enc.eot_token,
        valid_vocab=enc.n_vocab,
    )
    texts = []
    for row in out.tolist():
        tokens = row[1:]  # drop the leading <|endoftext|>
        if enc.eot_token in tokens:  # story finished -> cut at the end marker
            tokens = tokens[: tokens.index(enc.eot_token)]
        texts.append(enc.decode(tokens))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", default=None, help="default: ./model.pt or ./weights/model.pt"
    )
    parser.add_argument("--prompt", default=None, help="omit for interactive mode")
    parser.add_argument("--max_new_tokens", "-m", type=int, default=200)
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.8, help="0 = greedy"
    )
    parser.add_argument("--top_k", type=int, default=50, help="0 disables")
    parser.add_argument("--top_p", type=float, default=1.0, help="1.0 disables")
    parser.add_argument("--num_samples", "-n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    weights = find_weights(args.weights)
    model = load_model(weights, args.device)
    enc = tiktoken.get_encoding("gpt2")
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"loaded {weights} on {args.device} | {n_params:,} params | context {model.config.block_size}"
    )

    def run(prompt: str) -> None:
        for i, text in enumerate(
            generate_text(model, enc, prompt, args, args.device), 1
        ):
            if args.num_samples > 1:
                print(f"\n--- sample {i} ---")
            print(text)

    if args.prompt is not None:
        run(args.prompt)
        return

    print(
        "Interactive mode. Enter a prompt (empty line = unconditional story), Ctrl-D to quit."
    )
    while True:
        try:
            prompt = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        run(prompt)


if __name__ == "__main__":
    main()
