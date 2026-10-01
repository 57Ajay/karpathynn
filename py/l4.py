import torch.nn.functional as F
import random
import torch
import os
import matplotlib.pyplot as plt

with open("./data/names.txt", "r") as file:
    words = file.read().splitlines()

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

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


random.seed(69)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xva, Yva = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# Parameters
n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, n_embd), generator=g)
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)

W2 = torch.randn((n_hidden, 27), generator=g) * 0.01
b2 = torch.zeros(27)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# initial loss on first batch
emb = C[Xtr[:32]]
h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr[:32])

print(f"Initial loss with scaled W2: {loss.item():.4f}")
print(f"Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")

emb_sample = C[Xtr[:1000]]
z = emb_sample.view(-1, block_size * n_embd) @ W1 + b1
h = torch.tanh(z)

saturated_pct = (h.abs() > 0.99).float().mean().item() * 100
print(f"\n--- Tanh Health Check ---")
print(f"Pre-activation z: mean = {z.mean().item():.2f}, std = {z.std().item():.2f}")
print(f"Post-activation h: mean = {h.mean().item():.2f}, std = {h.std().item():.2f}")
print(f"Percentage of saturated units (|h| > 0.99): {saturated_pct:.2f}%")

# Ploting histograms and saturation heatmap
os.makedirs("plots", exist_ok=True)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Pre-activations z histogram
axs[0].hist(
    z.view(-1).tolist(), bins=50, density=True, color="skyblue", edgecolor="black"
)
axs[0].set_title(f"Pre-activations z (std: {z.std().item():.2f})")
axs[0].set_xlabel("Value")
axs[0].set_ylabel("Density")
axs[0].grid(True, alpha=0.3)

# 2. Hidden activations h histogram
axs[1].hist(
    h.view(-1).tolist(), bins=50, density=True, color="salmon", edgecolor="black"
)
axs[1].set_title("Activations h = tanh(z)")
axs[1].set_xlabel("Value")
axs[1].set_ylabel("Density")
axs[1].grid(True, alpha=0.3)

# 3. Saturation matrix (white = saturated |h| > 0.99)
# Display first 200 examples across all 200 neurons
axs[2].imshow(
    (h.abs() > 0.99)[:200].detach().cpu(), cmap="gray", interpolation="nearest"
)
axs[2].set_title("Saturation Map (White = |h| > 0.99)")
axs[2].set_xlabel("Neuron Index (0 to 199)")
axs[2].set_ylabel("Example Index (0 to 199)")

plt.tight_layout()
plt.savefig("plots/l04_tanh_saturation.png", dpi=120)
plt.close()
print("Saturation plot saved to plots/l04_tanh_saturation.png")
