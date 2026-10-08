use crate::model::Gpt;
use crate::rng::Rng;
use crate::tensor::{mm_nn, mm_nt};

pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
}

impl KvCache {
    pub fn new(model: &Gpt) -> KvCache {
        let cap = model.cfg.block_size * model.cfg.n_embd;
        KvCache {
            k: (0..model.cfg.n_layer)
                .map(|_| Vec::with_capacity(cap))
                .collect(),
            v: (0..model.cfg.n_layer)
                .map(|_| Vec::with_capacity(cap))
                .collect(),
            len: 0,
        }
    }
}

fn layer_norm_row(x: &[f32], g: &[f32], b: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let mu = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / n;
    let rs = 1.0 / (var + 1e-5).sqrt();
    for i in 0..x.len() {
        out[i] = g[i] * ((x[i] - mu) * rs) + b[i];
    }
}

fn gelu_inplace(x: &mut [f32]) {
    const K: f32 = 0.797_884_56;
    const A: f32 = 0.044715;
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + (K * (*v + A * *v * *v * *v)).tanh());
    }
}

fn linear_row(x: &[f32], w: &[f32], b: &[f32], nin: usize, nout: usize, out: &mut [f32]) {
    out.fill(0.0);
    mm_nn(x, w, 1, nin, nout, out);
    for i in 0..nout {
        out[i] += b[i];
    }
}

pub fn step(model: &Gpt, cache: &mut KvCache, token: u32, want_logits: bool) -> Option<Vec<f32>> {
    let cfg = &model.cfg;
    let (e, nh) = (cfg.n_embd, cfg.n_head);
    let hs = e / nh;
    let pos = cache.len;
    assert!(
        pos < cfg.block_size,
        "context is full (block_size {})",
        cfg.block_size
    );

    let mut x = vec![0.0f32; e];
    {
        let wte = model.wte.w.data();
        let wpe = model.wpe.w.data();
        for i in 0..e {
            x[i] = wte[token as usize * e + i] + wpe[pos * e + i];
        }
    }

    let mut xn = vec![0.0f32; e];
    let mut qkv = vec![0.0f32; 3 * e];
    let mut att_out = vec![0.0f32; e];
    let mut h1 = vec![0.0f32; 4 * e];
    let mut h2 = vec![0.0f32; e];

    for (l, blk) in model.blocks.iter().enumerate() {
        layer_norm_row(&x, &blk.ln_1.g.data(), &blk.ln_1.b.data(), &mut xn);
        linear_row(
            &xn,
            &blk.c_attn.w.data(),
            &blk.c_attn.b.as_ref().unwrap().data(),
            e,
            3 * e,
            &mut qkv,
        );
        let (q, kv) = qkv.split_at(e);
        let (k_new, v_new) = kv.split_at(e);
        cache.k[l].extend_from_slice(k_new);
        cache.v[l].extend_from_slice(v_new);
        let t = pos + 1;

        let kc = &cache.k[l];
        let vc = &cache.v[l];
        let scale = 1.0 / (hs as f32).sqrt();
        for h in 0..nh {
            let qo = h * hs;

            let mut scores = vec![0.0f32; t];
            for ti in 0..t {
                let krow = &kc[ti * e + qo..][..hs];
                let mut s = 0.0;
                for j in 0..hs {
                    s += q[qo + j] * krow[j];
                }
                scores[ti] = s * scale;
            }
            let mx = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut se = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                se += *s;
            }
            let inv = 1.0 / se;
            let out = &mut att_out[qo..qo + hs];
            out.fill(0.0);
            for ti in 0..t {
                let p = scores[ti] * inv;
                let vrow = &vc[ti * e + qo..][..hs];
                for j in 0..hs {
                    out[j] += p * vrow[j];
                }
            }
        }
        linear_row(
            &att_out,
            &blk.c_proj.w.data(),
            &blk.c_proj.b.as_ref().unwrap().data(),
            e,
            e,
            &mut h2,
        );
        for i in 0..e {
            x[i] += h2[i];
        }

        layer_norm_row(&x, &blk.ln_2.g.data(), &blk.ln_2.b.data(), &mut xn);
        linear_row(
            &xn,
            &blk.c_fc.w.data(),
            &blk.c_fc.b.as_ref().unwrap().data(),
            e,
            4 * e,
            &mut h1,
        );
        gelu_inplace(&mut h1);
        linear_row(
            &h1,
            &blk.mlp_proj.w.data(),
            &blk.mlp_proj.b.as_ref().unwrap().data(),
            4 * e,
            e,
            &mut h2,
        );
        for i in 0..e {
            x[i] += h2[i];
        }
    }
    cache.len += 1;

    if !want_logits {
        return None;
    }
    layer_norm_row(
        &x.clone(),
        &model.ln_f.g.data(),
        &model.ln_f.b.data(),
        &mut x,
    );
    let mut logits = vec![0.0f32; cfg.vocab_size];
    mm_nt(&x, &model.wte.w.data(), 1, e, cfg.vocab_size, &mut logits);
    Some(logits)
}

pub fn sample_logits(logits: &[f32], rng: &mut Rng, temperature: f32, top_k: Option<usize>) -> u32 {
    let mut l: Vec<f32> = logits.iter().map(|&x| x / temperature.max(1e-6)).collect();
    if let Some(k) = top_k {
        if k < l.len() {
            let mut sorted = l.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let thresh = sorted[k - 1];
            for x in l.iter_mut() {
                if *x < thresh {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
    }
    let mx = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let probs: Vec<f32> = l.iter().map(|&x| (x - mx).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let mut u = rng.uniform() * sum;
    for (i, &p) in probs.iter().enumerate() {
        u -= p;
        if u <= 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

fn ingest(model: &Gpt, cache: &mut KvCache, toks: &[u32]) -> Vec<f32> {
    let mut logits = None;
    for (i, &t) in toks.iter().enumerate() {
        logits = step(model, cache, t, i == toks.len() - 1);
    }
    logits.unwrap()
}

pub fn generate(
    model: &Gpt,
    prompt: &[u32],
    n_tokens: usize,
    rng: &mut Rng,
    temperature: f32,
    top_k: Option<usize>,
    mut on_token: impl FnMut(u32),
) -> Vec<u32> {
    assert!(!prompt.is_empty(), "prompt must contain at least one token");
    let block = model.cfg.block_size;
    let mut out = prompt.to_vec();
    let mut cache = KvCache::new(model);
    let start = out.len().saturating_sub(block - 1);
    let mut logits = ingest(model, &mut cache, &out[start..]);
    for _ in 0..n_tokens {
        let next = sample_logits(&logits, rng, temperature, top_k);
        on_token(next);
        out.push(next);
        if cache.len + 1 > block {
            let keep = (block / 2).max(1);
            cache = KvCache::new(model);
            logits = ingest(model, &mut cache, &out[out.len() - keep..]);
        } else {
            logits = step(model, &mut cache, next, true).unwrap();
        }
    }
    out
}
