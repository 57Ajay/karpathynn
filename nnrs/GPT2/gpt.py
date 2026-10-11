import math
import os
import inspect
from dataclasses import dataclass

import tiktoken
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

device_type = "cuda" if device.startswith("cuda") else "cpu"
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_head = config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        setattr(self.c_proj, "GPT_SCALE_INIT", 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(C, dim=-1)

        head_size = C // self.n_head

        # [B, T, C] -> [B, n_head, T, head_size]
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        setattr(self.c_proj, "GPT_SCALE_INIT", 1)

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

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # crop to block_size, since wpe only has that many positions
            idx_cond = idx[:, -self.config.block_size :]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        device_type: str,
        verbose: bool = True,
    ) -> torch.optim.AdamW:
        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        no_decay_params = [p for p in params if p.dim() < 2]

        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        if verbose:
            print(f"decayed:     {len(decay_params)} tensors, {n_decay:,} parameters")
            print(
                f"non-decayed: {len(no_decay_params)} tensors, {n_no_decay:,} parameters"
            )

        use_fused = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and device_type == "cuda"
        )
        if verbose:
            print(f"using fused AdamW: {use_fused}")

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def print_names_params_or_bufs(m: nn.Module, param: bool = True) -> None:
    print("\n---------------------------------------------------------")
    if param:
        print("--- parameters ---")
        for n, p in m.named_parameters():
            print(f"{n:32} {str(tuple(p.shape)):24} {p.numel():>12,}")
    else:
        print("--- buffers ---")
        for n, b in m.named_buffers():
            print(f"{n:32} {str(tuple(b.shape)):24} {b.numel():>12,}")
    print("---------------------------------------------------------\n")


class DataLoaderLite:
    def __init__(
        self,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
        file_path: str = "data/input.txt",
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        if len(self.tokens) < B * T * num_processes + 1:
            raise ValueError(
                f"Not enough tokens ({len(self.tokens)}) for "
                f"B*T*num_processes+1 = {B * T * num_processes + 1}"
            )

        if process_rank == 0:
            print(f"loaded {len(self.tokens):,} tokens")
            print(f"1 epoch = {len(self.tokens) // (B * T * num_processes):,} batches")

        # each rank starts at its own offset, so ranks read disjoint slices
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)  # targets, shifted by one

        self.current_position += B * T * self.num_processes

        # if the next batch would run past the end, wrap around
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = B * T * self.process_rank

        return x, y


def get_lr(
    step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int
) -> float:
    # linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # floor after the schedule ends
    if step > max_steps:
        return min_lr

    # cosine decay from max_lr down to min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


B, T = 4, 32  # micro-batch that fits in memory
total_batch_size = 512  # tokens per optimizer step
assert total_batch_size % (B * T * ddp_world_size) == 0, (
    "total_batch_size must be divisible by B * T * ddp_world_size"
)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(
        f"total batch: {total_batch_size:,} tokens | "
        f"world size: {ddp_world_size} | grad accum steps: {grad_accum_steps}"
    )

train_config = GPTConfig(
    block_size=T,
    vocab_size=50304,  # divisible by 64
    n_layer=4,
    n_head=4,
    n_embd=64,
)

train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size
)

model = GPT(train_config)
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(
        model, device_ids=[ddp_local_rank] if torch.cuda.is_available() else None
    )
raw_model = model.module if ddp else model

if master_process:
    print(f"params: {count_params(raw_model):,}")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 100
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=max_lr,
    device_type=device_type,
    verbose=master_process,
)

amp_enabled = device_type == "cpu" or torch.cuda.is_bf16_supported()

for step in range(max_steps):
    optimizer.zero_grad()
    loss_accum = torch.zeros((), device=device)

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        if ddp:
            # only all-reduce grads on the last micro-step
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            logits, loss = model(x, y)

        assert loss is not None
        # divide so the accumulated grads equal the full-batch mean
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        # SUM then divide: ReduceOp.AVG is nccl-only, gloo fails
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
        loss_accum /= ddp_world_size

    # clip once, on the fully accumulated grads
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )  # returns the pre-clip norm

    lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()

    if master_process and (step % 10 == 0 or step == max_steps - 1):
        print(
            f"step {step:4d} | loss {loss_accum.item():.4f} | lr {lr:.2e} | norm {norm.item():.4f}"
        )

if ddp:
    destroy_process_group()
