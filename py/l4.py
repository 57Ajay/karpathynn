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

fan_in = block_size * n_embd
gain = 5 / 3

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, n_embd), generator=g)

W1 = torch.randn((fan_in, n_hidden), generator=g) * torch.tensor(gain / fan_in**0.5)

# here itis batchnorm
# BatchNorm learnable parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

# we do not need this bias as it automaticallu gets cancelled
# when we subtraqct the mean in BatchNorm
# z - μ   = (x@W + b) - mean(x@W + b) = (x@W + b) - (mean(x@W) + b) = x@W - mean(x@W)
#      B

b1 = torch.zeros(n_hidden) * 0

# output layer
W2 = torch.randn((n_hidden, 27), generator=g) * 0.01
b2 = torch.zeros(27)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# forward passthrough batchnorm
emb = C[Xtr[:32]]
hpreact = emb.view(-1, fan_in) @ W1

# A->  batch mean and variance across dimension 0 (the batch dim)
bnmean = hpreact.mean(0, keepdim=True)
bnvar = hpreact.var(0, keepdim=True, unbiased=True)

# B -> normalize to standard normal
hpreact_norm = (hpreact - bnmean) / torch.sqrt(bnvar + 1e-5)

# C: scale and shift with learnable parameters
hpreact_final = bngain * hpreact_norm + bnbias

# D: activation
h = torch.tanh(hpreact_final)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr[:32])

print(
    f"hpreact_final: mean = {hpreact_final.mean().item():.4f}, std = {
        hpreact_final.std().item():.4f}"
)
print(f"Initial loss with BatchNorm: {loss.item():.4f}")


# emb_sample = C[Xtr[:1000]]
# z = emb_sample.view(-1, block_size * n_embd) @ W1 + b1
# h = torch.tanh(z)
#
# saturated_pct = (h.abs() > 0.99).float().mean().item() * 100
# print("\n--- Tanh Health Check ---")
# print(f"Pre-activation z: mean = {z.mean().item():.2f}, std = {z.std().item():.2f}")
# print(f"Post-activation h: mean = {h.mean().item():.2f}, std = {h.std().item():.2f}")
# print(f"Percentage of saturated units (|h| > 0.99): {saturated_pct:.2f}%")
#
# # Ploting histograms and saturation heatmap
# os.makedirs("plots", exist_ok=True)
# fig, axs = plt.subplots(1, 3, figsize=(18, 5))
#
# # 1. Pre-activations z histogram
# axs[0].hist(
#     z.view(-1).tolist(), bins=50, density=True, color="skyblue", edgecolor="black"
# )
# axs[0].set_title(f"Pre-activations z (std: {z.std().item():.2f})")
# axs[0].set_xlabel("Value")
# axs[0].set_ylabel("Density")
# axs[0].grid(True, alpha=0.3)
#
# # 2. Hidden activations h histogram
# axs[1].hist(
#     h.view(-1).tolist(), bins=50, density=True, color="salmon", edgecolor="black"
# )
# axs[1].set_title("Activations h = tanh(z)")
# axs[1].set_xlabel("Value")
# axs[1].set_ylabel("Density")
# axs[1].grid(True, alpha=0.3)
#
# # 3. Saturation matrix (white = saturated |h| > 0.99)
# # Display first 200 examples across all 200 neurons
# axs[2].imshow(
#     (h.abs() > 0.99)[:200].detach().cpu(), cmap="gray", interpolation="nearest"
# )
# axs[2].set_title("Saturation Map (White = |h| > 0.99)")
# axs[2].set_xlabel("Neuron Index (0 to 199)")
# axs[2].set_ylabel("Example Index (0 to 199)")
#
# plt.tight_layout()
# plt.savefig("plots/l04_kaiming_init.png", dpi=120)
# plt.close()
# print("Saturation plot saved to plots/l04_tanh_saturation.png")


# pythorchifying
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming normal init
        self.weight = torch.randn((fan_in, fan_out), generator=g) / (fan_in**0.5)
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Buffers (running statistics, updated with EMA without gradients)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=False)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# using above pythorchified way to train MLP
n_embd = 10
n_hidden = 100
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((27, n_embd), generator=g)

# Linear layers before BatchNorm use bias=False!
layers = [
    Linear(block_size * n_embd, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, 27, bias=True),
]

# Scaling the output layer to prevent hockey-stick loss
with torch.no_grad():
    layers[-1].weight *= 0.01

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(
    f"\nTotal parameters in modular deep model: {sum(p.nelement() for p in parameters)}"
)
for p in parameters:
    p.requires_grad = True

# --- TRAINING LOOP (200,000 STEPS) ---
max_steps = 200000
batch_size = 32

for i in range(max_steps):
    # Minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # Forward pass
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update with LR decay
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += torch.tensor(-lr) * p.grad

    if i % 20000 == 0 or i == max_steps - 1:
        print(f"step {i:6d} | lr: {lr:.2f} | minibatch loss: {loss.item():.4f}")


# --- EVALUATION (SWITCH TO EVAL MODE!) ---
for layer in layers:
    layer.training = False


@torch.no_grad()
def evaluate_split(X_split, Y_split):
    emb = C[X_split]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    return F.cross_entropy(x, Y_split).item()


train_loss = evaluate_split(Xtr, Ytr)
val_loss = evaluate_split(Xva, Yva)

print("\n--- Final Modular Evaluation ---")
print(f"Train loss: {train_loss:.4f}")
print(f"Val loss:   {val_loss:.4f}")

# --- SAMPLE NAMES (with training=False) ---
print("\n--- Generated Names (Modular Deep MLP with BatchNorm) ---")
g_sample = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        x = emb.view(1, -1)
        for layer in layers:
            x = layer(x)
        probs = F.softmax(x, dim=1)
        ix = int(torch.multinomial(probs, num_samples=1, generator=g_sample).item())
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
