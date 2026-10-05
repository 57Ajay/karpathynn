import random
import torch
import torch.nn.functional as F


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
