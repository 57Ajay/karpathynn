import itertools
from itertools import pairwise
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

with open("./data/names.txt", "r") as file:
    words = file.read().splitlines()


chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}


# b = {}
# N = torch.zeros((27, 27), dtype=torch.int32)
#
# for w in words:
#     chs = ["."] + list(w) + ["."]
#     for ch1, ch2 in pairwise(chs):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         N[ix1, ix2] += 1
#
# print("\n--- Tensor Stats ---")
# print(f"Tensor shape: {N.shape}")
# print(f"Total bigram counts: {N.sum().item()}")
# print(f"Names started (row 0 sum): {N[0].sum().item()}")
# print(f"Names ended (col 0 sum): {N[:, 0].sum().item()}")
#
# os.makedirs("plots", exist_ok=True)
# plt.figure(figsize=(16, 16))
# plt.imshow(N, cmap="Blues")
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color="gray", fontsize=8)
#         plt.text(
#             j, i, f"{N[i, j].item()}", ha="center", va="top", color="gray", fontsize=8
#         )
# plt.axis("off")
# plt.title("Bigram Counts Matrix (27x27)", fontsize=16)
# plt.savefig("plots/l02_bigram_counts.png", dpi=150, bbox_inches="tight")
# plt.close()
# print("Plot saved to plots/l02_bigram_counts.png")


# print("\nTop 10 bigrams:")
# for bigram, count in sorted(b.items(), key=lambda kv: -kv[1])[:10]:
#     print(f"{bigram}: {count}")
#
# P = (N + 1).float()
# P /= P.sum(1, keepdim=True)
#
# print(f"\nRow 0 sum: {P[0].sum().item():.4f}")
# print(f"Row 5 sum: {P[5].sum().item():.4f}")

g = torch.Generator().manual_seed(2147483647)
#
# for i in range(5):
#     out = []
#     ix: int = 0
#
#     while True:
#         p = P[ix]
#         ix = int(
#             torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#         )
#         out.append(itos[ix])
#         if ix == 0:
#             break
#
#     print("".join(out))

# log_likelihood = torch.tensor(0.0)
# n = 0
#
# for w in words:
#     chs = ["."] + list(w) + ["."]
#     for ch1, ch2 in pairwise(chs):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1, ix2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         n += 1
#
# nll = -log_likelihood
# avg_nll = nll / n
#
# print(f"log_likelihood: {log_likelihood.item()}, nll: {nll.item()}, avg_nll: {avg_nll}")
#
# test_word = "andrejq"
# print(f"\nEvaluating '{test_word}':")
# test_log_likelihood = torch.tensor(0.0)
# chs = ["."] + list(test_word) + ["."]
# for ch1, ch2 in itertools.pairwise(chs):
#     ix1, ix2 = stoi[ch1], stoi[ch2]
#     prob = P[ix1, ix2]
#     logprob = torch.log(prob)
#     test_log_likelihood += logprob
#     print(f"{ch1}{ch2}: prob={prob.item():.4f}, logprob={logprob.item():.4f}")
# print(f"Avg NLL for '{test_word}': {-test_log_likelihood.item() / len(test_word):.4f}")
#

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
# --------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------- #
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

xs, ys = [], []

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in itertools.pairwise(chs):
        xs.append(stoi[ch1])

        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

#
# print("\n--- Neural Net Dataset ---")
# print(f"Total training examples: {num}")
# print(f"First 5 inputs (xs): {xs[:5]}")
# print(f"First 5 targets (ys): {ys[:5]}")
#
xenc = F.one_hot(xs, num_classes=27).float()
# print(f"xenc shape: {xenc.shape}, dtype: {xenc.dtype}")

W = torch.randn((27, 27), generator=g, requires_grad=True)

# logits = xenc @ W
# counts = logits.exp()
# probs = counts / counts.sum(1, keepdim=True)
#
# print(f"\nlogits shape: {logits.shape}")
# print(f"probs shape: {probs.shape}")
# print(f"Row 0 sum of probs: {probs[0].sum().item():.4f}")

lr = 50.0
loss = torch.tensor(0.0)

for k in range(100):
    # forward pass
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean()

    if k % 10 == 0 or k == 99:
        print(f"step {k:3d} | loss: {loss.item():.4f}")

    # zero grad
    W.grad = None

    # backword pass
    loss.backward()

    # update
    W.data += -lr * W.grad

print(f"\nFinal loss after 100 steps: {loss.item():.4f}")

print("\n--- Embeddings Verification ---")
print(f"one-hot @ W for first example:\n{logits[0][:5]}")
print(f"Direct row lookup W[xs[0]]:\n{W[xs[0]][:5]}")
print(f"Are they identical? {torch.allclose(logits[0], W[xs[0]], atol=1e-5)}")

print("\n--- Generated Names (Neural Net) ---")
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        x_logits = W[ix]  # this here is more efficient than one-hot
        x_counts = x_logits.exp()
        p = x_counts / x_counts.sum()

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
