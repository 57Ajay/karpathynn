import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

with open("./data/names.txt", "r") as file:
    words = file.read().splitlines()

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}


block_size = 8
n_embd = 10
n_hidden = 68


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

    def __call__(self, x: torch.Tensor):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=False)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar.squeeze()

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


g = torch.Generator().manual_seed(69)


class Embedding:
    def __init__(self, vocab_size, n_embd):
        self.weight = torch.randn(
            (vocab_size, n_embd), generator=g
        )  # [vocab_size, n_embd]

    def __call__(self, IX):  # IX: [B, T]
        self.out = self.weight[IX]
        return self.out  # [B, T, n_embd]

    def parameters(self):
        return [self.weight]


# emb_layer = Embedding(vocab_size=27, n_embd=10)
# test_x = Xtr[:4]  # shape: (4, 8)
# test_out = emb_layer(test_x)
# print("Input shape: ", test_x.shape)
# print("Output shape:", test_out.shape)
# print("Num params:  ", sum(p.numel() for p in emb_layer.parameters()))


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)

        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out

    def parameters(self):
        return []


test_out = torch.randn((4, 8, 10))
flat8 = FlattenConsecutive(n=8)
flat2 = FlattenConsecutive(n=4)
#
# print("Input shape:         ", test_out.shape)
# print("With n=8 (flat MLP): ", flat8(test_out).shape)
# print("With n=2 (pairs):    ", flat2(test_out).shape)


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layers in self.layers:
            x = layers(x)
        self.x = x
        return self.x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


model = Sequential(
    [
        Embedding(27, n_embd),
        # Level 1: 8 tokens -> 4 pairs
        FlattenConsecutive(2),
        Linear(n_embd * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        # Level 2: 4 pairs -> 2 quads
        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        # Level 3: 2 quads -> 1 octet (squeezed to 2D)
        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        # Final classifier
        Linear(n_hidden, 27),
    ]
)


# Xb, Yb = Xtr[:4], Ytr[:4]
# x = Xb
# print("Input shape:", x.shape)
# for layer in model.layers:
#     x = layer(x)
#     print(f"{layer.__class__.__name__:20s} -> {tuple(x.shape)}")

with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters()
print(f"Total parameters: {sum(p.numel() for p in parameters)}")

for p in parameters:
    p.requires_grad = True

lossi = []

max_steps = 200000
batch_size = 32

for i in range(max_steps):
    # Minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # Forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    lossi.append(loss.log10().item())
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

for layer in model.layers:
    layer.training = False


@torch.no_grad()
def evaluate_split(X_split, Y_split):
    logits = model(X_split)
    return F.cross_entropy(logits, Y_split).item()


train_loss = evaluate_split(Xtr, Ytr)
val_loss = evaluate_split(Xva, Yva)

print(f"Train loss: {train_loss:.4f}")
print(f"Val loss:   {val_loss:.4f}")

for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = int(torch.multinomial(probs, num_samples=1).item())
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))


os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(10, 4))
# Reshaping 200,000 steps into (200, 1000) and then the mean across each 1,000 steps
smoothed_loss = torch.tensor(lossi).view(-1, 1000).mean(1)
plt.plot(smoothed_loss)
plt.title("L6 Baseline Flat Model Loss (Averaged over 1000-step windows)")
plt.xlabel("1k steps")
plt.ylabel("log10(loss)")
plt.grid(True)
plt.savefig("plots/l06_baseline_loss.png", dpi=120, bbox_inches="tight")
print("Saved loss plot to plots/l06_baseline_loss.png")
