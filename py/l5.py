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

# bnvar_inv = (bnvar + 1e-5)**-0.5 #(1, 64)
dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
cmp("bnvar", dbnvar, bnvar)

# bnvar = 1/(n - 1) * (bndiffsquarred).sum(0, keepdim=True) # (1, 64)
dbndiffsquarred = (1.0 / (n - 1)) * torch.ones_like(bndiffsquarred) * dbnvar
cmp("bndiffsquarred", dbndiffsquarred, bndiffsquarred)

# bndiffsquarred = bndiff**2 # (32, 64)
dbndiff += (2 * bndiff) * dbndiffsquarred
cmp("bndiff", dbndiff, bndiff)

# bndiff = hprebn - bnmeani # (32, 64)
dbnmeani = (-dbndiff).sum(0, keepdim=True)
cmp("bnmeani", dbnmeani, bnmeani)
dhprebn = dbndiff.clone()

# bnmeani = 1/n * hprebn.sum(0, keepdim=True) # (1, 64)
dhprebn += (1.0 / n) * torch.ones_like(hprebn) * dbnmeani
cmp("hprebn", dhprebn, hprebn)

# hprebn = embcat @ W1 + b1
dW1 = embcat.T @ dhprebn
cmp("W1", dW1, W1)
dembcat = dhprebn @ W1.T
cmp("embcat", dembcat, embcat)
db1 = dhprebn.sum(0)
cmp("b1", db1, b1)

# embcat = emb.view(emb.shape[0], -1)
demb = dembcat.view(emb.shape)
cmp("emb", demb, emb)

# emb = C[Xb]
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]
cmp("C", dC, C)

dlogits = F.softmax(logits, dim=1)
dlogits[range(n), Yb] -= 1.0
dlogits /= n
cmp("logits (fused)", dlogits, logits)


dhprebn = (bngain * bnvar_inv / n) * (
    n * dhpreact - dhpreact.sum(0) - (n / (n - 1)) * bnraw * (dhpreact * bnraw).sum(0)
)
cmp("hprebn (fused)", dhprebn, hprebn)


n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / ((n_embd * block_size) ** 0.5)
)
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1
bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = False

max_steps = 200000
batch_size = 32
n = batch_size
lossi = []

print(f"\nStarting 200K steps of training without autograd...")

with torch.no_grad():
    for i in range(max_steps):
        # 1. Minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # 2. Forward pass
        emb = C[Xb]  # (n, 3, 10)
        embcat = emb.view(emb.shape[0], -1)  # (n, 30)
        hprebn = embcat @ W1 + b1  # (n, 200)

        # BatchNorm forward
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5) ** -0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias

        # Non-linearity & Output
        h = torch.tanh(hpreact)  # (n, 200)
        logits = h @ W2 + b2  # (n, 27)
        loss = F.cross_entropy(logits, Yb)

        # 3. Pure Manual Backward Pass
        # Fused softmax + cross-entropy
        dlogits = F.softmax(logits, dim=1)
        dlogits[range(n), Yb] -= 1.0
        dlogits /= n

        # Layer 2 backprop
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)

        # Tanh backprop
        dhpreact = (1.0 - h**2) * dh

        # BatchNorm backprop
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = (bngain * bnvar_inv / n) * (
            n * dhpreact
            - dhpreact.sum(0)
            - (n / (n - 1)) * bnraw * (dhpreact * bnraw).sum(0)
        )

        # Layer 1 backprop
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)

        # Embedding backprop (vectorized accumulation)
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        dC.index_add_(0, Xb.view(-1), demb.view(-1, n_embd))

        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

        # 4. Parameter update (with learning rate decay)
        lr = 0.1 if i < 100000 else 0.01
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad

        # 5. Logging
        if i % 10000 == 0 or i == max_steps - 1:
            print(f"step {i:6d} / {max_steps:6d} | loss: {loss.item():.4f}")
        lossi.append(loss.item())

# Final Evaluation on Train & Validation sets
with torch.no_grad():
    for name, split_X, split_Y in [("train", Xtr, Ytr), ("val", Xva, Yva)]:
        emb = C[split_X]
        embcat = emb.view(emb.shape[0], -1)
        hprebn = embcat @ W1 + b1
        # Using batch statistics across the split
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5) ** -0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, split_Y)
        print(f"Final {name:5s} loss: {loss.item():.4f}")
