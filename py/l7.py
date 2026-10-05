import torch
import torch.nn.functional as F
import torch.nn as nn

with open("data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# unique chars and vocab_size
chars = sorted(set(text))
vocab_size = len(chars)

# chars mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(f"Total characters: {len(text):,}")
print(f"Vocab size: {vocab_size} characters: {''.join(chars)}")

# splitting data into training and validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4  # B
block_size = 8  # T


def get_batch(split):
    data_split = train_data if split == "train" else val_data

    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    x = torch.stack([data_split[i : i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])

    return x, y


xb, yb = get_batch("train")
print("Inputs shape (B, T): ", xb.shape)
print("Targets shape (B, T):", yb.shape)

print("\n--- training targets from 1 sequence chunk ---")
for t in range(block_size):
    context = xb[0, : t + 1].tolist()
    target = yb[0, t].item()
    print(
        f"When context is {decode(context)!r:20s} ---> target is {decode([target])!r}"
    )


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # [B, T, C]
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits_last = logits[:, -1, :]  # [B, C]
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel(vocab_size)
context = torch.zeros((1, 1), dtype=torch.long)
print("\n--- Untrained Model Generation (Gibberish) ---")
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32

for step in range(10000):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 2000 == 0 or step == 9999:
        print(f"step {step:5d} | loss: {loss.item():.4f}")

print("\n--- Trained Bigram Generation ---")
print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))
