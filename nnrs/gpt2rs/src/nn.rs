use crate::rng::Rng;
use crate::tensor::Tensor;

pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn new(rng: &mut Rng, nin: usize, nout: usize, bias: bool) -> Linear {
        let k = 1.0 / (nin as f32).sqrt();
        Linear {
            w: Tensor::rand_uniform(rng, &[nin, nout], -k, k),
            b: if bias {
                Some(Tensor::rand_uniform(rng, &[nout], -k, k))
            } else {
                None
            },
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let y = x.matmul(&self.w);
        match &self.b {
            Some(b) => y.add_bcast(b),
            None => y,
        }
    }

    pub fn params(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        out.push((format!("{prefix}.w"), self.w.clone()));
        if let Some(b) = &self.b {
            out.push((format!("{prefix}.b"), b.clone()));
        }
    }
}

pub struct Embedding {
    pub w: Tensor,
}

impl Embedding {
    pub fn new(rng: &mut Rng, vocab: usize, dim: usize) -> Embedding {
        Embedding {
            w: Tensor::randn(rng, &[vocab, dim], 1.0),
        }
    }

    pub fn forward(&self, idx: &[u32], prefix_shape: &[usize]) -> Tensor {
        self.w.gather_rows(idx, prefix_shape)
    }
}

pub struct LayerNorm {
    pub g: Tensor,
    pub b: Tensor,
}

impl LayerNorm {
    pub fn new(dim: usize) -> LayerNorm {
        LayerNorm {
            g: Tensor::ones(&[dim]),
            b: Tensor::zeros(&[dim]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.layer_norm(&self.g, &self.b)
    }

    pub fn params(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        out.push((format!("{prefix}.g"), self.g.clone()));
        out.push((format!("{prefix}.b"), self.b.clone()));
    }
}
