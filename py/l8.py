import itertools
import os
import matplotlib.pyplot as plt
import regex as re

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
COMPILED_PATTERN = re.compile(GPT2_SPLIT_PATTERN)


def get_stats(
    ids: list[int], counts: dict[tuple[int, int], int] | None = None
) -> dict[tuple[int, int], int]:
    if counts is None:
        counts = {}
    for id1, id2 in itertools.pairwise(ids):
        pair = (id1, id2)
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    new_ids = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    def register_special_tokens(self, special_tokens: dict[str, int]):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = True,
    ) -> list[tuple[int, int, int]]:
        """Trains BPE merges over regex chunks up to the target `vocab_size`.

        Returns merge history: [(step, top_pair_freq, total_token_len), ...]
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256

        # pre-tokenizing the text into semantic chunks
        chunks = COMPILED_PATTERN.findall(text)
        chunk_ids: list[list[int]] = [list(ch.encode("utf-8")) for ch in chunks if ch]

        history = []

        if verbose:
            initial_len = sum(len(c) for c in chunk_ids)
            print(f"Training BPE: {len(chunk_ids)} chunks, {initial_len} raw bytes")

        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for c in chunk_ids:
                get_stats(c, stats)

            if not stats:
                break

            top_pair = max(stats, key=stats.get)
            top_count = stats[top_pair]
            idx = 256 + i

            # Replacing pair in each chunk
            chunk_ids = [merge(c, top_pair, idx) for c in chunk_ids]

            # Recording merge & build vocabulary entry
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            current_len = sum(len(c) for c in chunk_ids)
            history.append((i + 1, top_count, current_len))

            if verbose and (
                (i + 1) % 50 == 0 or (i + 1) <= 10 or (i + 1) == num_merges
            ):
                token_repr = self.vocab[idx].decode("utf-8", errors="replace")
                print(
                    f"Merge {i + 1:3d}/{num_merges}: {top_pair} -> {idx} |"
                    f" {token_repr!r:12s} (count: {top_count:5d}, tokens:"
                    f" {current_len})"
                )

        return history

    def _encode_chunk(self, chunk_ids: list[int]) -> list[int]:
        """Encodes a single chunk by replaying learned merges in chronological order."""
        while len(chunk_ids) >= 2:
            stats = get_stats(chunk_ids)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            chunk_ids = merge(chunk_ids, pair, self.merges[pair])
        return chunk_ids

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Encodes arbitrary text into token IDs with regex chunks and special tokens."""
        if allowed_special is None:
            allowed_special = set()

        # Split on special tokens if present
        if not self.special_tokens or not allowed_special:
            chunks = COMPILED_PATTERN.findall(text)
            tokens = []
            for ch in chunks:
                tokens.extend(self._encode_chunk(list(ch.encode("utf-8"))))
            return tokens

        # Handle special tokens with regex partition
        special_pattern = "(" + "|".join(re.escape(k) for k in allowed_special) + ")"
        parts = re.split(special_pattern, text)
        tokens = []
        for part in parts:
            if part in allowed_special:
                tokens.append(self.special_tokens[part])
            elif part:
                chunks = COMPILED_PATTERN.findall(part)
                for ch in chunks:
                    tokens.extend(self._encode_chunk(list(ch.encode("utf-8"))))
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs back to a UTF-8 string."""
        byte_parts = []
        for idx in ids:
            if idx in self.vocab:
                byte_parts.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                byte_parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Unknown token ID: {idx}")
        return b"".join(byte_parts).decode("utf-8", errors="replace")


# Training on Tiny Shakespeare & Verification
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Training on first 40,000 characters for fast, crisp demonstration
    train_slice = text[:40000]
    target_vocab_size = 512  # 256 base bytes + 256 learned merges

    tokenizer = BPETokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": target_vocab_size})

    print("=" * 60)
    print(f"Training BPE Tokenizer (Vocab Size: {target_vocab_size})...")
    print("=" * 60)
    history = tokenizer.train(train_slice, vocab_size=target_vocab_size)

    # Lossless Roundtrip & Compression Benchmark
    test_sample = """MENENIUS:
There was a time when all the body's members
Rebelled against the belly, thus accused it:
<|endoftext|>
Hello world! Numbers: 12345 + 6789. Non-English: 안녕하세요! 🚀"""

    encoded = tokenizer.encode(test_sample, allowed_special={"<|endoftext|>"})
    decoded = tokenizer.decode(encoded)

    raw_bytes_len = len(test_sample.encode("utf-8"))
    token_len = len(encoded)
    compression_ratio = raw_bytes_len / token_len

    print("\n" + "=" * 60)
    print("ROUNDTRIP & COMPRESSION EVALUATION")
    print("=" * 60)
    print(f"Raw UTF-8 byte length : {raw_bytes_len} bytes")
    print(f"BPE Token length       : {token_len} tokens")
    print(f"Compression Ratio      : {compression_ratio:.2f}x")
    print(f"Lossless Roundtrip OK  : {decoded == test_sample}")
    assert decoded == test_sample, "Roundtrip verification failed!"

    # Plotting Compression Curve
    steps = [h[0] for h in history]
    token_lens = [h[2] for h in history]
    top_freqs = [h[1] for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    color = "tab:blue"
    ax1.set_xlabel("Merge Step (Number of Learned Tokens)", fontsize=11)
    ax1.set_ylabel("Total Corpus Length (Tokens)", color=color, fontsize=11)
    ax1.plot(steps, token_lens, color=color, linewidth=2, label="Token Length")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Merged Pair Frequency", color=color, fontsize=11)
    ax2.plot(
        steps,
        top_freqs,
        color=color,
        linewidth=1.5,
        linestyle="--",
        label="Top Pair Frequency",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(
        f"BPE Training: Sequence Length & Merge Frequency vs Iterations (Vocab:"
        f" {target_vocab_size})",
        fontsize=12,
    )
    fig.tight_layout()
    plot_path = "plots/l08_bpe_compression.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nSaved compression curve plot to: {plot_path}")
