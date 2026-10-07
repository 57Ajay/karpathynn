# Technical Architecture & Implementation Details: `minebprs`

This document details the internal design, algorithms, and technical trade-offs of `minebprs`.

---

## 1. The Byte Pair Encoding (BPE) Algorithm

BPE is a data compression algorithm adapted for subword language modeling:
1. Initialize the vocabulary with the 256 individual byte values ($0 \dots 255$).
2. Scan the corpus to compute adjacent token pair frequencies:
   $$\text{Stats}(p) = \sum_{i=0}^{N-2} \mathbb{I}[(\text{tok}_i, \text{tok}_{i+1}) == p]$$
3. Identify the most frequent pair $p^* = \arg\max_{p} \text{Stats}(p)$. In case of frequency ties, preserve insertion order to maintain parity with Python dictionary hashing semantics.
4. Mint a new token index:
   $$\text{ID}_{\text{new}} = 256 + r$$
   where $r$ is the 0-indexed merge iteration rank.
5. Substitute all non-overlapping adjacent occurrences of $p^*$ with $\text{ID}_{\text{new}}$.
6. Repeat until the target vocabulary size is reached.

---

## 2. Zero-Dependency Regex Scanner (`src/split.rs`)

### Motivation
Standard BPE trained directly on raw text tends to merge across semantic boundaries (e.g. `dog.` or `end\n`). Production LLMs (GPT-2, GPT-4, Llama) constrain merges to remain strictly within chunk boundaries.

### GPT-2 Pattern
```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

### Hand-Crafted Scanner
Instead of depending on third-party regular expression crates, `minebprs` implements a deterministic scanner over `char_indices`:
- Contraction check: identifies `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`.
- Leading space + Unicode Letter run (`char::is_alphabetic() && !char::is_numeric()`).
- Leading space + Unicode Digit run (`char::is_numeric()`).
- Leading space + Punctuation run (`!is_ws && !is_letter && !is_number`).
- Whitespace lookahead scanner (`\s+(?!\S)` vs trailing whitespace).

---

## 3. OpenAI Byte-Shuffle & `vocab.bpe` Inversion (`src/gpt2.rs`)

### OpenAI Printable Mapping (`bytes_to_unicode`)
OpenAI mapped the 256 byte values to printable Unicode characters so merge tables could be stored in standard text files:
- 188 bytes map to their ASCII / Latin-1 equivalent characters.
- The remaining 68 bytes are shifted to codepoints starting at 256 ($0x0100 \dots$).

### Byte Re-Indexing
In OpenAI's token vocabulary, token IDs $0 \dots 255$ do NOT correspond to raw byte values $0 \dots 255$. Instead, they correspond to the printable characters sorted by Unicode codepoint.
`Gpt2Tokenizer` computes:
```rust
let mut byte_to_id = [0u32; 256];
```
This maps incoming byte streams into shuffled token IDs prior to executing the 50,000 merge lookups.

---

## 4. Special Token Handling (`src/rgx.rs`)

Special tokens such as `<|endoftext|>` must never be fragmented into constituent subwords or merged into adjacent text.
- Special tokens are identified via substring scanning prior to chunk-level BPE encoding.
- Intervening text slices are routed through chunk splitting and BPE merging, while special tokens emit fixed out-of-band IDs.
- Support for `AllowedSpecial::All`, `AllowedSpecial::None`, `AllowedSpecial::NoneRaise`, and `AllowedSpecial::Set`.
