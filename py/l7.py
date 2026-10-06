import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size = 32
block_size = 8
n_embd = 32

torch.manual_seed(69)
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    x = torch.stack([data_split[i : i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


xb, yb = get_batch("train")


class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x: torch.Tensor):
        # X -> [B, T, C]
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weight = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, -float("inf"))
        weight = F.softmax(weight, dim=-1)

        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.sa_heads = MultiHeadAttention(4, n_embd // 4)  # self attention
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(
            n_embd, vocab_size
        )  # language model head, maps 32D vectors into 65 vocab logits

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None] | tuple[torch.Tensor, torch.Tensor]:
        # idx -> [B, T]
        _, T = idx.shape
        tok_emb: torch.Tensor = self.token_embedding_table(idx)
        pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb  # [B, T, n_embd]

        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        # idx -> [B, T]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            last_idx = logits[:, -1, :]
            probs = F.softmax(last_idx, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, sample), dim=1)

        return idx


model = BigramLanguageModel(vocab_size)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

eval_iters = 200


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


optemizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(10000):
    if i % 500 == 0:
        losses = estimate_loss()
        print(
            f"step {i:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optemizer.zero_grad(set_to_none=True)
    loss.backward()
    optemizer.step()


print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))
"""
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

xbow1 = torch.zeros((B, T, C))

for b in range(B):
    for t in range(T):
        tok = x[b, : t + 1]
        mean = tok.mean(dim=0)
        xbow1[b, t] = mean


tril_mat = torch.tril(torch.ones(T, T))
weight = tril_mat / tril_mat.sum(dim=1, keepdim=True)

xbow2 = weight @ x

print(torch.allclose(xbow1, xbow2))

weight = torch.zeros((T, T))
weight = weight.masked_fill(tril_mat == 0, -float("inf"))
weight = F.softmax(weight, dim=-1)
xbow3 = weight @ x

print(torch.allclose(xbow2, xbow3))

head_size = C

k = torch.randn((B, T, head_size))
q = torch.randn((B, T, head_size))
v = torch.randn((B, T, head_size))
weight = q @ k.transpose(-2, -1)
weight /= head_size**0.5
weight = weight.masked_fill(tril_mat == 0, -float("inf"))
weight = F.softmax(weight, dim=-1)
xbow4 = weight @ v

print(torch.allclose(xbow4, xbow3))
"""
