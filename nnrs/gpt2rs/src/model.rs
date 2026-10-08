use crate::nn::{Embedding, LayerNorm, Linear};
use crate::rng::Rng;
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub dropout: f32,
}

impl GptConfig {
    pub fn gpt2_124m() -> GptConfig {
        GptConfig {
            vocab_size: 50257,
            block_size: 1024,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            dropout: 0.0,
        }
    }

    pub fn mini(vocab_size: usize) -> GptConfig {
        GptConfig {
            vocab_size,
            block_size: 64,
            n_layer: 4,
            n_head: 4,
            n_embd: 128,
            dropout: 0.0,
        }
    }
}

pub struct Block {
    pub ln_1: LayerNorm,
    pub c_attn: Linear,
    pub c_proj: Linear,
    pub ln_2: LayerNorm,
    pub c_fc: Linear,
    pub mlp_proj: Linear,
    n_head: usize,
}

fn linear_n(rng: &mut Rng, nin: usize, nout: usize, std: f32) -> Linear {
    Linear {
        w: Tensor::randn(rng, &[nin, nout], std),
        b: Some(Tensor::zeros(&[nout])),
    }
}

impl Block {
    fn new(cfg: &GptConfig, rng: &mut Rng) -> Block {
        let e = cfg.n_embd;

        let proj_std = 0.02 / ((2 * cfg.n_layer) as f32).sqrt();
        Block {
            ln_1: LayerNorm::new(e),
            c_attn: linear_n(rng, e, 3 * e, 0.02),
            c_proj: linear_n(rng, e, e, proj_std),
            ln_2: LayerNorm::new(e),
            c_fc: linear_n(rng, e, 4 * e, 0.02),
            mlp_proj: linear_n(rng, 4 * e, e, proj_std),
            n_head: cfg.n_head,
        }
    }

    fn forward(&self, x: &Tensor, b: usize, t: usize, p: f32, rng: &mut Rng) -> Tensor {
        let e = *x.shape().last().unwrap();
        let nh = self.n_head;
        let hs = e / nh;

        let xn = self.ln_1.forward(x);
        let qkv = self.c_attn.forward(&xn);
        let split_head = |z: Tensor| {
            z.reshape(&[b, t, nh, hs])
                .transpose12()
                .reshape(&[b * nh, t, hs])
        };
        let q = split_head(qkv.slice_last(0, e));
        let k = split_head(qkv.slice_last(e, e));
        let v = split_head(qkv.slice_last(2 * e, e));
        let att = q
            .bmm_t(&k)
            .scale(1.0 / (hs as f32).sqrt())
            .causal_softmax()
            .dropout(p, rng);
        let y = att
            .bmm(&v)
            .reshape(&[b, nh, t, hs])
            .transpose12()
            .reshape(&[b, t, e]);
        let x = x.add(&self.c_proj.forward(&y).dropout(p, rng));

        let xn2 = self.ln_2.forward(&x);
        let m = self.mlp_proj.forward(&self.c_fc.forward(&xn2).gelu());
        x.add(&m.dropout(p, rng))
    }
}

pub struct Gpt {
    pub cfg: GptConfig,
    pub wte: Embedding,
    pub wpe: Embedding,
    pub blocks: Vec<Block>,
    pub ln_f: LayerNorm,
}

impl Gpt {
    pub fn new(cfg: &GptConfig, rng: &mut Rng) -> Gpt {
        Gpt {
            cfg: *cfg,
            wte: Embedding {
                w: Tensor::randn(rng, &[cfg.vocab_size, cfg.n_embd], 0.02),
            },
            wpe: Embedding {
                w: Tensor::randn(rng, &[cfg.block_size, cfg.n_embd], 0.02),
            },
            blocks: (0..cfg.n_layer).map(|_| Block::new(cfg, rng)).collect(),
            ln_f: LayerNorm::new(cfg.n_embd),
        }
    }

    pub fn forward(&self, idx: &[u32], b: usize, t: usize, p: f32, rng: &mut Rng) -> Tensor {
        assert!(t <= self.cfg.block_size);
        let tok = self.wte.forward(idx, &[b, t]);
        let pos_ids: Vec<u32> = (0..t as u32).collect();
        let pos = self.wpe.forward(&pos_ids, &[t]);
        let mut x = tok.add_bcast(&pos).dropout(p, rng);
        for blk in &self.blocks {
            x = blk.forward(&x, b, t, p, rng);
        }
        self.ln_f.forward(&x).matmul_t(&self.wte.w)
    }

    pub fn loss(
        &self,
        idx: &[u32],
        targets: &[i32],
        b: usize,
        t: usize,
        p: f32,
        rng: &mut Rng,
    ) -> Tensor {
        self.forward(idx, b, t, p, rng).cross_entropy(targets)
    }

    pub fn params(&self) -> Vec<(String, Tensor)> {
        let mut p = vec![
            ("wte.weight".to_string(), self.wte.w.clone()),
            ("wpe.weight".to_string(), self.wpe.w.clone()),
        ];
        let lin = |name: String, l: &Linear, out: &mut Vec<(String, Tensor)>| {
            out.push((format!("{name}.weight"), l.w.clone()));
            out.push((format!("{name}.bias"), l.b.as_ref().unwrap().clone()));
        };
        for (i, blk) in self.blocks.iter().enumerate() {
            p.push((format!("h.{i}.ln_1.weight"), blk.ln_1.g.clone()));
            p.push((format!("h.{i}.ln_1.bias"), blk.ln_1.b.clone()));
            lin(format!("h.{i}.attn.c_attn"), &blk.c_attn, &mut p);
            lin(format!("h.{i}.attn.c_proj"), &blk.c_proj, &mut p);
            p.push((format!("h.{i}.ln_2.weight"), blk.ln_2.g.clone()));
            p.push((format!("h.{i}.ln_2.bias"), blk.ln_2.b.clone()));
            lin(format!("h.{i}.mlp.c_fc"), &blk.c_fc, &mut p);
            lin(format!("h.{i}.mlp.c_proj"), &blk.mlp_proj, &mut p);
        }
        p.push(("ln_f.weight".to_string(), self.ln_f.g.clone()));
        p.push(("ln_f.bias".to_string(), self.ln_f.b.clone()));
        p
    }

    pub fn num_params(&self) -> usize {
        self.params().iter().map(|(_, t)| t.numel()).sum()
    }
}
