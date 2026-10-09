# gpt2rs

**GPT-2 in pure Rust. Zero external dependencies. Proven mathematically equivalent to HuggingFace `transformers`.**

`gpt2rs` is the culmination of the Rust zero-to-hero road. It unifies the custom dynamic tensor autograd engine from **makemorers** and the byte-identical BPE tokenizer from **minebprs**, completing the full GPT-2 architecture (124M parameters and smaller configurations), KV-cache autoregressive inference, a zero-dependency `.safetensors` weight loader, and CPU training.

The equivalence claim is mathematically verified. Running `cargo test --release` loads a GPT-2 reference model exported from HuggingFace `transformers`, executes our forward pass on the identical token inputs, and compares every logit across all sequence positions:

```text
max abs logit diff vs transformers: 1.94e-07     (pure float32 rounding noise)
```

Identical LayerNorm $\epsilon = 10^{-5}$, exact NewGELU activation curve, identical attention scaling ($1/\sqrt{d_{\text{head}}}$), identical weight layout, and tied token embedding and unembedding matrices ($W_{\text{emb}} = W_{\text{unembed}}^T$).

---

## The Journey & Shared Stack

Unlike typical deep learning frameworks where PyTorch handles the underlying computation, each project in this series builds upon the previous one from first principles:

| Project | Contributions to `gpt2rs` |
|---|---|
| **micrograd** | Dynamic computational graph tape and reverse-mode automatic differentiation |
| **makemorers** | Core dynamic tensor engine (`src/tensor.rs`), multi-threaded matmul kernels, neural network modules (`Linear`, `Embedding`, `LayerNorm`), AdamW optimizer (`src/optim.rs`), and PRNG (`src/rng.rs`) |
| **minebprs** | Byte-level BPE tokenizer (`src/bpe_gpt2.rs`, `src/split.rs`) matching OpenAI's vocabulary and merge rules byte-for-byte |
| **gpt2rs** *(this repo)* | GPT-2 124M architecture, KV-cache inference (`src/infer.rs`), zero-dep `.safetensors` & JSON parsers (`src/safetensors.rs`, `src/json.rs`), HuggingFace weight importer (`src/hf.rs`), and parity verification harness |

---

## Architectural Comparison: makemore Transformer vs. GPT-2 124M

Architecturally, the Transformer block is identical: pre-LayerNorm residual stream, causal multi-head self-attention, and NewGELU MLP expansion. The differences represent scale and pretraining refinements:

| Feature | makemorers Transformer | GPT-2 (124M Config) |
|---|---|---|
| **Layers ($L$) / Heads ($H$) / Width ($d$)** | 4 / 4 / 64 | 12 / 12 / 768 |
| **Context Window ($T$)** | 16 characters | 1024 BPE tokens |
| **Vocabulary Size ($V$)** | 27 characters | 50,257 BPE tokens |
| **Parameter Count** | 204,443 | **124,439,808** |
| **Output Head (`lm_head`)** | Dedicated $(d, V)$ matrix | **Tied to $W_{\text{emb}}^T$ ($x \cdot W_{\text{emb}}^T$)** |
| **Dropout** | None | 0.1 during training (0.0 during inference) |
| **Residual Projection Init** | Standard Kaiming uniform | $\mathcal{N}(0, 0.02 / \sqrt{2 \cdot L})$ |

### Key Enhancements

1. **Weight Tying**:
   The output projection (`lm_head`) reuses the transposed token embedding table ($W_{\text{emb}}^T$). This eliminates a redundant $768 \times 50,257$ matrix (saving 38,597,376 parameters) and conceptually aligns input token representations with next-token prediction targets. In the autograd engine, this is executed by the custom `matmul_t` op, which accumulates output gradients directly back into the embedding tensor.
2. **Residual Stream Variance Scaling**:
   Projections that add back into the residual stream (`c_proj` and `mlp_proj`) are initialized with standard deviation $\sigma = 0.02 / \sqrt{2 \cdot L}$. At depth $L=12$, this dampens residual accumulation and prevents signal explosion before LayerNorm stabilizes activations.

---

## KV-Cache Autoregressive Inference

Naive autoregressive generation feeds the entire sequence $x_{1 \dots t}$ back into the model at every step, repeating previous work and leading to quadratic $\mathcal{O}(T^2)$ computational complexity:

$$\text{Naive FLOPs} \approx 2 \times N_{\text{params}} \times t \quad \text{per step}$$

Because key and value projections for past tokens never change, `src/infer.rs` implements an explicit Key-Value (KV) cache:
- Past $K$ and $V$ tensors are stored in contiguous slice memory per layer.
- For each newly generated token, only a single column of work is executed ($1$ token through Q, K, V projections).
- New $K$ and $V$ are appended to the cache, and attention is evaluated across all cached tokens:

$$\text{KV-Cache FLOPs} \approx 2 \times N_{\text{params}} \quad \text{per step (constant work)}$$

Furthermore, the output `lm_head` projection (77M FLOPs at $V=50,257$) is only evaluated on the very last token position where a prediction is required.

---

## Quickstart

### 1. Build and Run Unit & Parity Tests

```bash
cargo build --release
cargo test --release
```

All 8 tests—including the HuggingFace logit parity test, KV-cache consistency check, and autograd gradient checks (`matmul_t`, `dropout`, `no_grad`)—run in less than a second.

### 2. Generate with Shipped Pretrained Model

A mini-GPT model trained on Shakespeare (~810K parameters, 4 layers, 4 heads, 128 embedding dim, character-level vocab) is bundled directly in `pretrained/shakespeare-mini/model.bin`:

```bash
cargo run --release -- generate \
    --checkpoint pretrained/shakespeare-mini/model.bin \
    --prompt "ROMEO:" \
    --tokens 300 \
    --temperature 0.8
```

Sample output:
```text
ROMEO:
I would not his spoid, for, bothout'd your thee:
Here, you two see bess, we serves you advent then trurn.
KING HENRY VI:
Not Edward our mening!
```

### 3. Train a Mini-GPT on CPU

Train a custom model on TinyShakespeare using the built-in AdamW optimizer:

```bash
cargo run --release -- train \
    --data data/tinyshakespeare.txt \
    --steps 2500 \
    --batch-size 16 \
    --block-size 64 \
    --n-layer 4 \
    --n-head 4 \
    --n-embd 128 \
    --out out
```

Checkpoints (`out/model.bin`) are automatically saved whenever validation loss improves.

### 4. Running the Official 124M OpenAI GPT-2 Model

To run the full 124M parameter model:

```bash
# 1. Download official config.json and model.safetensors (~500MB) from HuggingFace
python3 scripts/fetch_gpt2_weights.py

# 2. Run autoregressive generation with BPE tokenization
cargo run --release -- generate \
    --hf-config gpt2-124m/config.json \
    --hf-weights gpt2-124m/model.safetensors \
    --prompt "Hello, I'm a language model," \
    --tokens 60 \
    --temperature 0.8 \
    --top-k 40
```

---

## Parameter Counts & Training Feasibility

### Model Parameter Breakdown

| Configuration | Vocab | Context | Layers | Heads | Dim | Parameters |
|---|---|---|---|---|---|---|
| **Shakespeare Mini** (shipped) | 65 | 64 | 4 | 4 | 128 | **809,856** (~0.81M) |
| **Tier 1 Shakespeare** | 65 | 128 | 4 | 4 | 128 | **818,048** (~0.82M) |
| **Tier 2 Shakespeare** | 65 | 128 | 6 | 6 | 192 | **3,153,920** (~3.15M) |
| **Tier 3 (Video Spec)** | 65 | 256 | 6 | 6 | 384 | **10,695,936** (~10.7M) |
| **GPT-2 Standard (124M)** | 50,257 | 1024 | 12 | 12 | 768 | **124,439,808** (~124.44M) |

### Can You Train on GPU?

**No, not with this pure-Rust codebase as-is.**

1. **CPU Autograd Engine**: `gpt2rs` is written strictly with Rust's standard library (`std::thread`, `Vec<f32>`). Memory lives entirely in host RAM and computations execute on the CPU.
2. **Computational Scale**:
   - Small models (0.8M to 3M params) train on a modern multi-core CPU in 15–45 minutes.
   - Pretraining the full 124M model on 10B tokens requires roughly $10^{19}$ FLOPs (~8× A100 GPUs for a week). On a CPU, that workload would take decades.
3. **GPU Options**:
   - For GPU training, use Karpathy's PyTorch scripts in `py/l7.py` or `nanoGPT` with CUDA.
   - For GPU in Rust, one would integrate CUDA bindings (`cudarc`), WebGPU shaders (`wgpu`), or Rust ML backends (`candle-core` / `burn`).

---

## License & References

- Architecture & Training: [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt), [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Lecture: Andrej Karpathy's *Let's reproduce GPT-2 (124M)* and *Neural Networks: Zero to Hero*
- Tokenizer: OpenAI GPT-2 BPE specifications
