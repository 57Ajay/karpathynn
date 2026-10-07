use crate::tensor::Tensor;

pub struct AdamW {
    params: Vec<Tensor>,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub lr: f32,
    pub weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: i32,
}

impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> AdamW {
        let m = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        AdamW {
            params,
            m,
            v,
            lr,
            weight_decay,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            t: 0,
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        for (i, p) in self.params.iter().enumerate() {
            let g = p.grad();
            let mut d = p.data_mut();
            let (m, v) = (&mut self.m[i], &mut self.v[i]);
            for j in 0..d.len() {
                let gj = g[j];
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * gj;
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * gj * gj;
                let mhat = m[j] / bc1;
                let vhat = v[j] / bc2;
                d[j] -= self.lr * (mhat / (vhat.sqrt() + self.eps) + self.weight_decay * d[j]);
            }
        }
    }
}
