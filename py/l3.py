from matplotlib.pylab import require
from numpy import dtype
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random

with open("./data/names.txt", "r") as file:
    words = file.read().splitlines()

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3

X, Y = [], []

for w in words:
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

print(
    f"X shape: {X.shape}\nX dtype: {X.dtype}\nY shape: {Y.shape}\nY dtype: {Y, dtype}"
)

# print("\nFirst 5 (X -> Y) examples:")
# for x, y in zip(X[:5], Y[:5]):
#     context_str = "".join(itos[int(i.item())] for i in x)
#     target_str = itos[int(y.item())]
#     print(f"'{context_str}' ({x.tolist()}) ---> '{target_str}' ({y.item()})")
#

# g = torch.Generator().manual_seed(2147483647)
#
# # this is feature embedding Matrix
# C = torch.randn((27, 2), generator=g, requires_grad=True)
#
# # this one is lookup embedding
# emb = C[X]
#
# print(f"\nEmbedding tensor shape: {emb.shape}")
# print(f"Example 0 context embedding:\n{emb[0]}")
#
# # this here is hiddle layer
# W1 = torch.randn((6, 100), generator=g, requires_grad=True)
# b1 = torch.randn(100, generator=g, requires_grad=True)
#
# # this is output later
#
# W2 = torch.randn((100, 27), generator=g, requires_grad=True)
# b2 = torch.randn(27, generator=g, requires_grad=True)
#
# parameters = [C, W1, b1, W2, b2]
#
# print(f"total parameters:{sum(p.nelement() for p in parameters)}")
#
# # forward pass
# h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
#
# logits = h @ W2 + b2
#
# loss = F.cross_entropy(logits, Y)
#
# print(f"h shape: {h.shape}")
# print(f"logits shape: {logits.shape}")
# print(f"Initial loss: {loss.item():.4f}")
#
# # ok let's overfiut to check if model architecture and autograd is ok
# for k in range(50):
#     emb_batch = C[X[:32]]
#     h_batch = torch.tanh(emb_batch.view(-1, 6) @ W1 + b1)
#     logits_batch = h_batch @ W2 + b2
#     loss_batch = F.cross_entropy(logits_batch, Y[:32])
#
#     for p in parameters:
#         p.grad = None
#
#     loss_batch.backward()
#
#     for p in parameters:
#         p.data += torch.tensor(-0.01) * p.grad
#
#     if k % 10 == 0 or k == 49:
#         print(f"step {k:2d} | single batch loss: {loss_batch.item():.4f}")
#
#
# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre

# these ae just for plotting so we can know best optimal learning rate for us
# lri = []
# lossi = []
#
# for i in range(1000):
#     ix = torch.randint(0, X.shape[0], (32,))
#
#     emb = C[X[ix]]  # 32, 3, 2
#     h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
#     logits = h @ W2 + b2  # 32, 27
#     loss = F.cross_entropy(logits, Y[ix])
#
#     for p in parameters:
#         p.grad = None
#     loss.backward()
#
#     lr = lrs[i]
#     for p in parameters:
#         p.data += -lr * p.grad
#
#     lri.append(lre[i].item())
#     lossi.append(loss.item())

# os.makedirs("plots", exist_ok=True)
# plt.figure(figsize=(8, 5))
# plt.plot(lri, lossi)
# plt.xlabel("Learning Rate Exponent (10^x)")
# plt.ylabel("Minibatch Loss")
# plt.title("Learning Rate Range Test (10^-3 to 10^0)")
# plt.grid(True, alpha=0.3)
# plt.savefig("plots/l03_lr_search.png", dpi=120, bbox_inches="tight")
# plt.close()
# print("\nLR Search completed! Plot saved to plots/l03_lr_search.png")
#
# now we scale it up and train properly
random.seed(69)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))


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


Xtr, Ytr = build_dataset(words[:n1])
Xva, Yva = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print("\n--- Data Splits ---")
print(f"Train: {Xtr.shape[0]} examples")
print(f"Val:   {Xva.shape[0]} examples")
print(f"Test:  {Xte.shape[0]} examples")

n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, n_embd), generator=g, requires_grad=True)
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g, requires_grad=True)
b1 = torch.randn(n_hidden, generator=g, requires_grad=True)
W2 = torch.randn((n_hidden, 27), generator=g, requires_grad=True)
b2 = torch.randn(27, generator=g, requires_grad=True)
parameters = [C, W1, b1, W2, b2]

print(f"\nTotal scaled parameters: {sum(p.nelement() for p in parameters)}")

lossi = []


for i in range(200000):
    # here is Minibatch
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emb = C[Xtr[ix]]

    h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = torch.tensor(0.1) if i < 100000 else torch.tensor(0.01)

    for p in parameters:
        p.data += -lr * p.grad

    if i % 20000 == 0 or i == 199999:
        print(f"step {i:6d} | lr: {lr:.2f} | minibatch loss: {loss.item():.4f}")


@torch.no_grad()
def evaluate_split(X_split, Y_split):
    emb = C[X_split]
    h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y_split).item()


train_loss = evaluate_split(Xtr, Ytr)
val_loss = evaluate_split(Xva, Yva)

print("\n--- Final Evaluation ---")
print(f"Train loss: {train_loss:.4f}")
print(f"Val loss:   {val_loss:.4f}")

print("\n--- Generated Names (Bengio MLP) ---")
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]  # (1, block_size, n_embd)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)

        ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
