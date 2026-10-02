from l4 import bnmean
from matplotlib.pylab import shape
import torch
import torch.nn.functional as F

with open("./data/names.txt", "r") as file:
    words = file.read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 3


def build_dataset(words_subset):
    X, Y = [], []
    for w in words_subset:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xva, Yva = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# Model initialization
n_embd = 10
n_hidden = 64

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)

# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * torch.tensor(
    (5 / 3) / ((n_embd * block_size) ** 0.5)
)
b1 = (
    torch.randn(n_hidden, generator=g) * 0.1
)  # including it to verify its gradient behavior

# Layer 2
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1

# BatchNorm parameters
bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

# Taking Fixed batch of 32 examples
n = 32
Xb, Yb = Xtr[:n], Ytr[:n]


# Atomic unrolled forward pass
emb = C[Xb]  # (n, 3, 10)
embcat = emb.view(emb.shape[0], -1)  # (n, 30)

# Linear 1
hprebn = embcat @ W1 + b1  # (n, 64)

# BatchNorm
bnmeani = 1 / n * hprebn.sum(0, keepdim=True)  # (1, 64)
bndiff = hprebn - bnmeani  # (n, 64)
bndiffsquarred = bndiff**2  # (n, 64)
bnvar = 1 / (n - 1) * (bndiffsquarred).sum(0, keepdim=True)  # Bessel's correction (n-1)
bnvar_inv = (bnvar + 1e-5) ** -0.5  # (1, 64)
bnraw = bndiff * bnvar_inv  # (n, 64)
hpreact = bngain * bnraw + bnbias  # (n, 64)

# Non-linearity
h = torch.tanh(hpreact)  # (n, 64)

# Linear 2
logits = h @ W2 + b2  # (n, 27)

# Cross-entropy loss unrolled into elementary ops
logit_maxes = logits.max(1, keepdim=True).values  # (n, 1)
norm_logits = logits - logit_maxes  # (n, 27)
counts = norm_logits.exp()  # (n, 27)
counts_sum = counts.sum(1, keepdim=True)  # (n, 1)
counts_sum_inv = counts_sum**-1  # (n, 1)
probs = counts * counts_sum_inv  # (n, 27)
logprobs = probs.log()  # (n, 27)
loss = -logprobs[range(n), Yb].mean()  # scalar

# PyTorch autograd reference
for p in parameters:
    p.grad = None
for t in [
    logprobs,
    probs,
    counts_sum_inv,
    counts_sum,
    counts,
    norm_logits,
    logit_maxes,
    logits,
    h,
    hpreact,
    bnraw,
    bnvar_inv,
    bnvar,
    bndiffsquarred,
    bndiff,
    bnmeani,
    hprebn,
    embcat,
    emb,
]:
    t.retain_grad()
loss.backward()


# Gradient comparison utility
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(
        f"{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}"
    )


print(f"Forward pass complete. Initial batch loss: {loss.item():.4f}")

# so n is 32 yb is 27 hence logprobs is 32 X 27 that is 864
# elements. now [range(n), yb] plucks out one column per row
# which leaves 26 others, hence there grad is 0
# therefore we have 32 non zero grads and 832 zero grads

# loss = -logprobs[range(n), Yb].mean()
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n  # (n, 27)
cmp("logprobs", dlogprobs, logprobs)

# logprobs = probs.log()
dprobs = (1 / probs) * dlogprobs  # (n, 27)
cmp("probs", dprobs, probs)

# probs = counts * counts_sum_inv

# now here we have counts of shape (n, 27) and
# counts_sum_inv of shape (n, 1) hence here we
# have broadcasting, where PyTorch will broadcast
# counts_sum_inv for all 27 columns, that means a
# single value in row(i) is influenced by all 27
# probablility in that row
# hebce we will sum-up all the incoming contributions
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)  # (n, 1)
cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)

# counts_sum_inv = counts_sum**-1
dcounts_sum = dcounts_sum_inv * (-(counts_sum ** (-2)))  # (n, 1)
cmp("dcounts_sum", dcounts_sum, counts_sum)

# counts_sum = counts.sum(1, keepdim=True)
# now here counts feed into 2 ops:
# probs = counts * counts_sum_inv  # (n, 27) and
# counts_sum = counts.sum(1, keepdim=True)  # (n, 1)
# 1st contribution of probs with count
dcounts = dprobs * counts_sum_inv  # (n, 27)
# 2nd contribution of counts_sum
dcounts += torch.ones_like(counts) * dcounts_sum
cmp("counts", dcounts, counts)

# counts = norm_logits.exp (n, 27)
dnorm_logits = dcounts * counts
cmp("norm_logits", dnorm_logits, norm_logits)

# norm_logits = logits - logit_maxes (n, 27)
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp("logit_maxes", dlogit_maxes, logit_maxes)

# logit_maxes = logits.max(1, keepdim=True).values (n, 1)
dlogits = dnorm_logits.clone()
dlogits += F.one_hot(logits.max(1).indices, num_classes=vocab_size) * dlogit_maxes
cmp("logits", dlogits, logits)

# logits = h @ W2 + b2
dh = dlogits @ W2.T
cmp("h", dh, h)
dW2 = h.T @ dlogits
cmp("W2", dW2, W2)
db2 = dlogits.sum(0)
cmp("b2", db2, b2)

# h = torch.tanh(hpreact)
dhpreact = (1 - h**2) * dh
cmp("hpreact", dhpreact, hpreact)

# hpreact = bngain * bnraw + bnbias
dbnraw = bngain * dhpreact  # (32, 64)
cmp("bnraw", dbnraw, bnraw)
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)  # (1, 64)
cmp("bngain", dbngain, bngain)
dbnbias = dhpreact.sum(0, keepdim=True)  # (1, 64)
cmp("bnbias", dbnbias, bnbias)

# bnraw = bndiff * bnvar_inv (n, 64)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
cmp("bnvar_inv", dbnvar_inv, bnvar_inv)

dbndiff = bnvar_inv * dbnraw
cmp("bndiff", dbndiff, bndiff)

# bnvar_inv = (bnvar + 1e-5)**-0.5 #(1, 64)
dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
cmp("bnvar", dbnvar, bnvar)

# bnvar = 1/(n - 1) * (bndiffsquarred).sum(0, keepdim=True) # (1, 64)
dbndiffsquarred = (1.0 / (n - 1)) * torch.ones_like(bndiffsquarred) * dbnvar
cmp("bndiffsquarred", dbndiffsquarred, bndiffsquarred)

# bndiffsquarred = bndiff**2 # (32, 64)
dbndiff += (2 * bndiff) + dbndiffsquarred
cmp("bndiff", dbndiff, bndiff)

# bndiff = hprebn - bnmeani # (32, 64)
dbnmeani = (-dbndiff).sum(0, keepdim=True)
cmp("bnmeani", dbnmeani, bnmean)
dhprebn = dbndiff.clone()

# bnmeani = 1/n * hprebn.sum(0, keepdim=True) # (1, 64)
dhprebn += (1.0 / n) * torch.ones_like(hprebn) * dbnmeani
cmp("hprebn", dhprebn, hprebn)
