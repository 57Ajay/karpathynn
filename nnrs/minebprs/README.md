# minebprs

A pure Rust implementation of the Byte Pair Encoding (BPE) tokenizer from Andrej Karpathy's *Neural Networks: Zero to Hero* series.

Zero external dependencies (`std` only). Produces token sequences byte-identical to OpenAI's reference GPT-2 tokenizer.

---

## Overview

Modern Large Language Models (LLMs) operate on discrete token sequences rather than raw Unicode strings or individual characters. `minebprs` provides a complete, high-performance, from-scratch implementation of BPE tokenization covering:

1. **Basic BPE Tokenizer (`BasicTokenizer`)**:
   - Operates directly over raw UTF-8 byte streams.
   - Iteratively discovers and merges the most frequent adjacent token pairs.
   - Exact tie-breaking matching Python dictionary insertion order.

2. **Regex Chunk Splitter (`SplitPattern`)**:
   - Zero-dependency scanner reproducing Python `regex` split patterns for GPT-2 and GPT-4.
   - Prevents merges across whitespace, punctuation, and contraction boundaries.

3. **Regex BPE Tokenizer (`RegexTokenizer`)**:
   - Applies BPE merging strictly within chunk boundaries.
   - Out-of-band special token handling (`<|endoftext|>`, etc.).

4. **Production GPT-2 Tokenizer (`Gpt2Tokenizer`)**:
   - Loads OpenAI's published 50,000-merge `vocab.bpe`.
   - Full printable Unicode byte-shuffle translation (`bytes_to_unicode`).
   - Produces exact token IDs matching OpenAI's `tiktoken` and `gpt-3-encoder` across all 50,257 tokens.

---

## Quick Start

### Build and Test

```bash
cargo build --release
cargo test
```

### CLI Usage

```bash
# Train BPE on custom text
cargo run --release -- train --input tests/taylorswift.txt --vocab-size 512 --type regex --output models/ts512

# Encode text using trained model
cargo run --release -- encode --model models/ts512.model --text "Hello world"

# Decode token IDs
cargo run --release -- decode --model models/ts512.model --ids "31373 995"

# Run official GPT-2 tokenizer with OpenAI merges
cargo run --release -- gpt2 --text "Hello, I'm a language model,"
# Output: 15496 11 314 1101 257 3303 2746 11
```

---

## Architecture

```
Raw Text (UTF-8)
      │
      ▼
┌─────────────────────────┐
│ Regex Chunking Scanner  │  (GPT-2 / GPT-4 split pattern)
└───────────┬─────────────┘
            │  Vector of chunks: ["Hello", "'m", " a", " language", ...]
            ▼
┌─────────────────────────┐
│ Pair Stats Accumulator  │  (Pair frequencies across chunks)
└───────────┬─────────────┘
            │  Rank-ordered merges
            ▼
┌─────────────────────────┐
│ Iterative Merging Loop  │  (Lowest-rank merge applied first)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Token ID Sequence       │  Vec<u32>
└─────────────────────────┘
```

---

## Verification & Parity

All test suites verify exact numerical equivalence:
- **`tests/bpe.rs`**: Wikipedia BPE example, roundtrips, serialization, and compression ratio.
- **`tests/split.rs`**: Scanner parity against Python `regex` outputs for contractions, unicode, and whitespace.
- **`tests/gpt2_parity.rs`**: Byte-for-byte token ID parity with OpenAI's GPT-2 tokenizer.
