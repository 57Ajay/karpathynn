from itertools import pairwise
import os
from os.path import join
import matplotlib.pyplot as plt
import torch

with open("./data/names.txt", "r") as file:
    words = file.read().splitlines()


chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

#
# print(f"Total words: {len(words)}")
# print(f"First 5 words: {words[:5]}")
# print(f"Min length: {min(len(w) for w in words)}")
# print(f"Max length: {max(len(w) for w in words)}")

b = {}
N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in pairwise(chs):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

print("\n--- Tensor Stats ---")
print(f"Tensor shape: {N.shape}")
print(f"Total bigram counts: {N.sum().item()}")
print(f"Names started (row 0 sum): {N[0].sum().item()}")
print(f"Names ended (col 0 sum): {N[:, 0].sum().item()}")

os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray", fontsize=8)
        plt.text(
            j, i, f"{N[i, j].item()}", ha="center", va="top", color="gray", fontsize=8
        )
plt.axis("off")
plt.title("Bigram Counts Matrix (27x27)", fontsize=16)
plt.savefig("plots/l02_bigram_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to plots/l02_bigram_counts.png")


# print("\nTop 10 bigrams:")
# for bigram, count in sorted(b.items(), key=lambda kv: -kv[1])[:10]:
#     print(f"{bigram}: {count}")
