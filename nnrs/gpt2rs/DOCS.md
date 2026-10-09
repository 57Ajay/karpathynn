# Technical Architecture & Implementation Deep-Dive: `gpt2rs`

This document details the mathematical derivations, internal architecture, memory layouts, and implementation nuances of `gpt2rs`.

---

## 1. Mathematical Formulation of the GPT-2 Architecture

GPT-2 is a decoder-only Transformer adopting the **Pre-LayerNorm** formulation (Wang et al., 2019 / Radford et al., 2019).

### 1.1 Input Representation

Given an input sequence of integer token indices $\mathbf{x} = [x_1, x_2, \dots, x_T] \in \mathbb{N}^T$:

1. **Token Embedding ($W_{\text{emb}} \in \mathbb{R}^{V \times d}$)**:
   $$\mathbf{e}_t = W_{\text{emb}}[x_t] \in \mathbb{R}^d$$
2. **Positional Embedding ($W_{\text{pos}} \in \mathbb{R}^{T_{\text{max}} \times d}$)**:
   $$\mathbf{p}_t = W_{\text{pos}}[t] \in \mathbb{R}^d, \quad 0 \le t < T$$
3. **Combined Input**:
   $$\mathbf{h}_0^{(t)} = \mathbf{e}_t + \mathbf{p}_t$$

### 1.2 Transformer Block (Pre-LN Residual Stream)

For each block $\ell \in \{1, \dots, L\}$:

$$\mathbf{h}_{\ell-1}' = \text{LayerNorm}(\mathbf{h}_{\ell-1})$$
$$\mathbf{a}_\ell = \text{CausalSelfAttention}(\mathbf{h}_{\ell-1}') + \mathbf{h}_{\ell-1}$$
$$\mathbf{a}_\ell' = \text{LayerNorm}(\mathbf{a}_\ell)$$
$$\mathbf{h}_\ell = \text{MLP}(\mathbf{a}_\ell') + \mathbf{a}_\ell$$

#### Layer Normalization
$$\text{LayerNorm}(\mathbf{z}) = \frac{\mathbf{z} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$
where $\epsilon = 10^{-5}$, and $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ are learnable affine parameters.

#### Causal Multi-Head Self-Attention
1. Linear QKV projection ($W_{\text{qkv}} \in \mathbb{R}^{d \times 3d}$ with bias $b_{\text{qkv}} \in \mathbb{R}^{3d}$):
   $$[Q, K, V] = \mathbf{z} W_{\text{qkv}} + b_{\text{qkv}}$$
   where $Q, K, V \in \mathbb{R}^{T \times d}$.
2. Split into $H$ heads of dimension $d_h = d / H$:
   $$Q_h, K_h, V_h \in \mathbb{R}^{T \times d_h}, \quad h \in \{1, \dots, H\}$$
3. Scaled dot-product attention with causal triangular masking:
   $$\text{Attn}(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_h}} + M\right) V_h$$
   $$M_{i,j} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$
4. Output projection ($W_{\text{proj}} \in \mathbb{R}^{d \times d}$ with bias $b_{\text{proj}} \in \mathbb{R}^d$):
   $$\mathbf{y} = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_{\text{proj}} + b_{\text{proj}}$$

#### MLP Sub-layer
GPT-2 uses the NewGELU non-linearity (an approximation of the Gaussian Error Linear Unit):

$$\text{NewGELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 \cdot x^3\right)\right)\right)$$

The MLP expands the hidden dimension by a factor of 4:
$$\text{MLP}(\mathbf{z}) = \text{NewGELU}(\mathbf{z} W_{\text{fc}} + b_{\text{fc}}) W_{\text{mlp\_proj}} + b_{\text{mlp\_proj}}$$
where $W_{\text{fc}} \in \mathbb{R}^{d \times 4d}$ and $W_{\text{mlp\_proj}} \in \mathbb{R}^{4d \times d}$.

### 1.3 Weight Tying & Output Head

After the final LayerNorm $\mathbf{h}_L' = \text{LayerNorm}_f(\mathbf{h}_L)$, logits over vocabulary $V$ are computed:

$$\mathbf{logits} = \mathbf{h}_L' \cdot W_{\text{emb}}^T \in \mathbb{R}^{T \times V}$$

Because the output projection shares memory with the transposed input embedding $W_{\text{emb}}^T$, no separate `lm_head.weight` is stored.

---

## 2. KV-Cache & Autoregressive Inference (`src/infer.rs`)

### 2.1 The Quadratic Bottleneck of Naive Decoding

In naive autoregressive decoding, generating token $T+1$ re-runs the entire prompt and previous history $[x_1, \dots, x_T]$. For a sequence of length $N$:
$$\text{Total FLOPs}_{\text{naive}} = \sum_{t=1}^N \mathcal{O}(t \cdot L \cdot d^2) = \mathcal{O}(N^2 \cdot L \cdot d^2)$$

### 2.2 KV-Cache Design

Because $K$ and $V$ vectors for tokens $1 \dots t-1$ depend solely on past representations, they can be cached:

```
Step t:
  x_t (single token)
   │
   ▼
┌─────────────────────────┐
│ LayerNorm & QKV Linear  │ ──► q_t (current query)
└───────────┬─────────────┘
            │
            ├─────────────► k_t ──► Append to Layer Cache: K_cached = [K_{1..t-1}, k_t]
            └─────────────► v_t ──► Append to Layer Cache: V_cached = [V_{1..t-1}, v_t]
                                      │
┌─────────────────────────┐           │
│ Attention:              │ ◄─────────┘
│ softmax(q_t · K^T / √d) │
│       · V               │
└───────────┬─────────────┘
            │
            ▼
    y_t (single output)
```

By maintaining a contiguous buffer per layer:
```rust
pub struct LayerCache {
    pub k: Vec<f32>, // shape (capacity, n_embd)
    pub v: Vec<f32>, // shape (capacity, n_embd)
    pub len: usize,
}
```
Each generation step executes only $\mathcal{O}(1 \cdot L \cdot d^2)$ projection work and $\mathcal{O}(t \cdot L \cdot d)$ attention vector products, reducing total generation complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.

---

## 3. Zero-Dependency Safetensors Loader (`src/safetensors.rs`)

### 3.1 Binary Format Layout

Safetensors is a simple, zero-copy machine learning tensor storage format:

```
┌────────────────────────────────────────────────────────┐
│ 8 bytes: Header Length N (u64, Little-Endian)          │
├────────────────────────────────────────────────────────┤
│ N bytes: UTF-8 JSON Header String                      │
│ {                                                      │
│   "wte.weight": {                                      │
│     "dtype": "F32",                                    │
│     "shape": [50257, 768],                             │
│     "data_offsets": [0, 154389504]                     │
│   }, ...                                               │
│ }                                                      │
├────────────────────────────────────────────────────────┤
│ Raw Tensor Byte Buffer (aligned float32 arrays)        │
│ [offset_start .. offset_end]                           │
└────────────────────────────────────────────────────────┘
```

### 3.2 Parser Implementation

`gpt2rs` reads the file without external JSON or deserialization crates:
1. Reads the first 8 bytes as `u64::from_le_bytes`.
2. Slices bytes `8 .. 8 + N` as a UTF-8 string and parses it with our recursive-descent parser (`src/json.rs`).
3. For each tensor:
   - Verifies `dtype == "F32"`.
   - Extracts `shape: Vec<usize>`.
   - Reads `[start, end]` slice from `(8 + N + start) .. (8 + N + end)`.
   - Converts byte slice into `Vec<f32>` using `f32::from_le_bytes`.

### 3.3 HuggingFace Weight Mapping (`src/hf.rs`)

HuggingFace's GPT-2 implementation (`transformers.GPT2LMHeadModel`) uses `Conv1D` for linear layers, storing weights with shape `(in_features, out_features)`. This aligns with our internal `Linear` layout $y = x W + b$, allowing direct parameter transfer without matrix transposition.

Prefix handling strips optional `transformer.` namespaces (`transformer.wte.weight` $\to$ `wte.weight`) and ignores precomputed causal mask buffers (`attn.bias`).

---

## 4. Zero-Dependency JSON Parser (`src/json.rs`)

`src/json.rs` provides a minimal recursive-descent tokenizer and parser supporting:
- JSON Objects (`{ "key": value }`)
- Arrays (`[1, 2, 3]`)
- Strings with escape handling
- Integers and floating-point numbers
- Booleans and null

This parser processes both the Safetensors header and `config.json` without pull-in dependencies like `serde` or `serde_json`.

---

## 5. Dynamic Autograd Extensions (`src/tensor.rs`)

### 5.1 Weight Tying Op: `matmul_t`

Standard matrix multiplication computes $Z = X W$. When the output head is tied to the embedding table, we need $Z = X W^T$, where $X \in \mathbb{R}^{B \cdot T \times d}$ and $W = W_{\text{emb}} \in \mathbb{R}^{V \times d}$.

In `tensor.rs`:
- Forward pass: $Z_{i,j} = \sum_k X_{i,k} W_{j,k}$
- Backward pass:
  $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W$$
  $$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Z}\right)^T X$$

During the backward traversal of the tape, gradients from both `wte.forward()` (the embedding gather) and `matmul_t` accumulate into `wte.w.grad`, automatically handling the weight sharing.

### 5.2 `no_grad` Mode

To conserve memory and prevent graph tape allocation during validation evaluation and autoregressive sampling:
```rust
pub fn no_grad() -> bool
pub fn set_no_grad(v: bool)
```
When `no_grad` is enabled, operations allocate only data buffers, bypassing tape node creation.

---

## 6. Mathematical Verification & Parity Proof

`tests/parity.rs` rigorously compares `gpt2rs` against HuggingFace `transformers` across all logits.

### Parity Invariant

$$\max_{t, v} | \text{logit}_{\text{gpt2rs}}(t, v) - \text{logit}_{\text{HF}}(t, v) | < 2 \times 10^{-6}$$

Observed maximum absolute difference:
$$\Delta_{\max} \approx 1.94 \times 10^{-7}$$

This confirms:
- Identical LayerNorm variance calculation ($\frac{1}{N} \sum (x - \mu)^2$).
- Identical GELU polynomial coefficients ($0.044715$).
- Causal triangular attention mask structure.
- Correct scaling factor $1/\sqrt{d_{\text{head}}}$.
- Exact weight layout and index alignments.

---

## 7. Complete Parameter Count Formulas

For any GPT-2 configuration defined by:
- $V$: Vocabulary size
- $T$: Context window (block size)
- $L$: Number of layers
- $H$: Number of attention heads
- $d$: Embedding dimension (model width)

### Parameter Equations

1. **Token Embeddings ($W_{\text{emb}}$)**:
   $$N_{\text{wte}} = V \times d$$
2. **Positional Embeddings ($W_{\text{pos}}$)**:
   $$N_{\text{wpe}} = T \times d$$
3. **Transformer Block (per block)**:
   - LN 1 ($\gamma, \beta$): $2d$
   - QKV Projection ($W_{\text{attn}}, b_{\text{attn}}$): $d \times (3d) + 3d = 3d^2 + 3d$
   - Attention Projection ($W_{\text{proj}}, b_{\text{proj}}$): $d \times d + d = d^2 + d$
   - LN 2 ($\gamma, \beta$): $2d$
   - MLP Expansion ($W_{\text{fc}}, b_{\text{fc}}$): $d \times (4d) + 4d = 4d^2 + 4d$
   - MLP Projection ($W_{\text{mlp\_proj}}, b_{\text{mlp\_proj}}$): $(4d) \times d + d = 4d^2 + d$
   - Total per block:
     $$N_{\text{block}} = 12d^2 + 13d$$
4. **Final LayerNorm ($W_{\text{ln\_f}}, b_{\text{ln\_f}}$)**:
   $$N_{\text{ln\_f}} = 2d$$
5. **Output Head (`lm_head`)**:
   $$N_{\text{head}} = 0 \quad (\text{Tied to } W_{\text{emb}}^T)$$

### Total Unique Parameter Count

$$N_{\text{total}} = (V + T) \cdot d + L \cdot (12d^2 + 13d) + 2d$$

#### For GPT-2 Standard (124M):
- $V = 50,257$
- $T = 1024$
- $L = 12$
- $d = 768$

Calculation:
- $N_{\text{wte}} = 50,257 \times 768 = 38,597,376$
- $N_{\text{wpe}} = 1024 \times 768 = 786,432$
- $N_{\text{block}} = 12 \times 768^2 + 13 \times 768 = 7,077,888 + 9,984 = 7,087,872$
- $12 \text{ blocks} = 12 \times 7,087,872 = 85,054,464$
- $N_{\text{ln\_f}} = 2 \times 768 = 1,536$
- **Total: $38,597,376 + 786,432 + 85,054,464 + 1,536 = \mathbf{124,439,808}$**
