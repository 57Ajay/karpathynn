use std::cell::{Cell, Ref, RefCell, RefMut};
use std::collections::HashSet;
use std::rc::Rc;

use crate::rng::Rng;

type BackwardFn = Box<dyn Fn(&[f32], &[f32])>;

thread_local! {
    static NO_GRAD: Cell<bool> = const { Cell::new(false) };
}

pub fn set_no_grad(on: bool) {
    NO_GRAD.with(|c| c.set(on));
}

pub fn is_no_grad() -> bool {
    NO_GRAD.with(|c| c.get())
}

pub struct Inner {
    data: RefCell<Vec<f32>>,
    grad: RefCell<Vec<f32>>,
    shape: Vec<usize>,
    prev: Vec<Tensor>,
    backward: Option<BackwardFn>,
}

pub struct Tensor(Rc<Inner>);

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor(Rc::clone(&self.0))
    }
}

impl Tensor {
    pub fn leaf(data: Vec<f32>, shape: &[usize]) -> Tensor {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data/shape mismatch"
        );
        let n = data.len();
        Tensor(Rc::new(Inner {
            data: RefCell::new(data),
            grad: RefCell::new(vec![0.0; n]),
            shape: shape.to_vec(),
            prev: Vec::new(),
            backward: None,
        }))
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::leaf(vec![0.0; shape.iter().product()], shape)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::leaf(vec![1.0; shape.iter().product()], shape)
    }

    pub fn randn(rng: &mut Rng, shape: &[usize], std: f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::leaf((0..n).map(|_| rng.normal() * std).collect(), shape)
    }

    pub fn rand_uniform(rng: &mut Rng, shape: &[usize], lo: f32, hi: f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::leaf(
            (0..n).map(|_| rng.uniform() * (hi - lo) + lo).collect(),
            shape,
        )
    }

    fn from_op(
        data: Vec<f32>,
        shape: Vec<usize>,
        prev: Vec<Tensor>,
        backward: BackwardFn,
    ) -> Tensor {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        if is_no_grad() {
            return Tensor::leaf(data, &shape);
        }
        let n = data.len();
        Tensor(Rc::new(Inner {
            data: RefCell::new(data),
            grad: RefCell::new(vec![0.0; n]),
            shape,
            prev,
            backward: Some(backward),
        }))
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn numel(&self) -> usize {
        self.0.shape.iter().product()
    }

    pub fn data(&self) -> Ref<'_, Vec<f32>> {
        self.0.data.borrow()
    }

    pub fn data_mut(&self) -> RefMut<'_, Vec<f32>> {
        self.0.data.borrow_mut()
    }

    pub fn grad(&self) -> Ref<'_, Vec<f32>> {
        self.0.grad.borrow()
    }

    fn grad_mut(&self) -> RefMut<'_, Vec<f32>> {
        self.0.grad.borrow_mut()
    }

    pub fn zero_grad(&self) {
        self.0.grad.borrow_mut().fill(0.0);
    }

    pub fn item(&self) -> f32 {
        self.0.data.borrow()[0]
    }

    pub fn backward(&self) {
        enum Frame {
            Enter(Tensor),
            Exit(Tensor),
        }
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<*const Inner> = HashSet::new();
        let mut stack = vec![Frame::Enter(self.clone())];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(t) => {
                    if !visited.insert(Rc::as_ptr(&t.0)) {
                        continue;
                    }
                    stack.push(Frame::Exit(t.clone()));
                    for p in &t.0.prev {
                        if !visited.contains(&Rc::as_ptr(&p.0)) {
                            stack.push(Frame::Enter(p.clone()));
                        }
                    }
                }
                Frame::Exit(t) => topo.push(t),
            }
        }
        self.0.grad.borrow_mut().fill(1.0);
        for t in topo.iter().rev() {
            if let Some(f) = &t.0.backward {
                let d = t.0.data.borrow();
                let g = t.0.grad.borrow();
                f(&d[..], &g[..]);
            }
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape());
        let data: Vec<f32> = {
            let a = self.data();
            let b = other.data();
            a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
        };
        let (ac, bc) = (self.clone(), other.clone());
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone(), other.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let mut g = ac.grad_mut();
                    for (gi, &o) in g.iter_mut().zip(go) {
                        *gi += o;
                    }
                }
                {
                    let mut g = bc.grad_mut();
                    for (gi, &o) in g.iter_mut().zip(go) {
                        *gi += o;
                    }
                }
            }),
        )
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape());
        let data: Vec<f32> = {
            let a = self.data();
            let b = other.data();
            a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
        };
        let (ac, bc) = (self.clone(), other.clone());
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone(), other.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let mut g = ac.grad_mut();
                    for (gi, &o) in g.iter_mut().zip(go) {
                        *gi += o;
                    }
                }
                {
                    let mut g = bc.grad_mut();
                    for (gi, &o) in g.iter_mut().zip(go) {
                        *gi -= o;
                    }
                }
            }),
        )
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape());
        let data: Vec<f32> = {
            let a = self.data();
            let b = other.data();
            a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
        };
        let (ac, bc) = (self.clone(), other.clone());
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone(), other.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let b = bc.data();
                    let mut g = ac.grad_mut();
                    for i in 0..go.len() {
                        g[i] += go[i] * b[i];
                    }
                }
                {
                    let a = ac.data();
                    let mut g = bc.grad_mut();
                    for i in 0..go.len() {
                        g[i] += go[i] * a[i];
                    }
                }
            }),
        )
    }

    pub fn scale(&self, c: f32) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&x| x * c).collect();
        let ac = self.clone();
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let mut g = ac.grad_mut();
                for (gi, &o) in g.iter_mut().zip(go) {
                    *gi += c * o;
                }
            }),
        )
    }

    pub fn tanh(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&x| x.tanh()).collect();
        let ac = self.clone();
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone()],
            Box::new(move |yd: &[f32], go: &[f32]| {
                let mut g = ac.grad_mut();
                for i in 0..go.len() {
                    g[i] += (1.0 - yd[i] * yd[i]) * go[i];
                }
            }),
        )
    }

    pub fn sigmoid(&self) -> Tensor {
        let data: Vec<f32> = self
            .data()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let ac = self.clone();
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone()],
            Box::new(move |yd: &[f32], go: &[f32]| {
                let mut g = ac.grad_mut();
                for i in 0..go.len() {
                    g[i] += yd[i] * (1.0 - yd[i]) * go[i];
                }
            }),
        )
    }

    pub fn gelu(&self) -> Tensor {
        const K: f32 = 0.797_884_56;
        const A: f32 = 0.044715;
        let data: Vec<f32> = self
            .data()
            .iter()
            .map(|&x| 0.5 * x * (1.0 + (K * (x + A * x * x * x)).tanh()))
            .collect();
        let ac = self.clone();
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let x = ac.data();
                let mut g = ac.grad_mut();
                for i in 0..go.len() {
                    let xi = x[i];
                    let t = (K * (xi + A * xi * xi * xi)).tanh();
                    let dy =
                        0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * K * (1.0 + 3.0 * A * xi * xi);
                    g[i] += dy * go[i];
                }
            }),
        )
    }

    pub fn sum(&self) -> Tensor {
        let s: f64 = self.data().iter().map(|&x| x as f64).sum();
        let ac = self.clone();
        Tensor::from_op(
            vec![s as f32],
            vec![1],
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let g0 = go[0];
                let mut g = ac.grad_mut();
                for gi in g.iter_mut() {
                    *gi += g0;
                }
            }),
        )
    }

    pub fn add_bcast(&self, other: &Tensor) -> Tensor {
        let an = self.numel();
        let bn = other.numel();
        assert!(
            an % bn == 0 && self.shape().ends_with(other.shape()),
            "broadcast shape mismatch"
        );
        let data: Vec<f32> = {
            let a = self.data();
            let b = other.data();
            (0..an).map(|i| a[i] + b[i % bn]).collect()
        };
        let (ac, bc) = (self.clone(), other.clone());
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone(), other.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let mut g = ac.grad_mut();
                    for (gi, &o) in g.iter_mut().zip(go) {
                        *gi += o;
                    }
                }
                {
                    let mut g = bc.grad_mut();
                    for (i, &o) in go.iter().enumerate() {
                        g[i % bn] += o;
                    }
                }
            }),
        )
    }

    pub fn repeat_rows(&self, reps: usize) -> Tensor {
        assert_eq!(self.shape()[0], 1);
        let c = self.numel();
        let data: Vec<f32> = {
            let a = self.data();
            (0..reps * c).map(|i| a[i % c]).collect()
        };
        let ac = self.clone();
        Tensor::from_op(
            data,
            vec![reps, c],
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let mut g = ac.grad_mut();
                for (i, &o) in go.iter().enumerate() {
                    g[i % c] += o;
                }
            }),
        )
    }

    pub fn matmul(&self, w: &Tensor) -> Tensor {
        let k = *self.shape().last().unwrap();
        assert_eq!(w.shape().len(), 2);
        assert_eq!(w.shape()[0], k);
        let n = w.shape()[1];
        let m = self.numel() / k;
        let mut out = vec![0.0; m * n];
        {
            let a = self.data();
            let b = w.data();
            mm_nn(&a[..], &b[..], m, k, n, &mut out);
        }
        let mut shape = self.shape().to_vec();
        *shape.last_mut().unwrap() = n;
        let (xc, wc) = (self.clone(), w.clone());
        Tensor::from_op(
            out,
            shape,
            vec![self.clone(), w.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let wd = wc.data();
                    let mut xg = xc.grad_mut();
                    mm_nt(go, &wd[..], m, n, k, &mut xg[..]);
                }
                {
                    let xd = xc.data();
                    let mut wg = wc.grad_mut();
                    mm_tn(&xd[..], go, m, k, n, &mut wg[..]);
                }
            }),
        )
    }

    pub fn matmul_t(&self, w: &Tensor) -> Tensor {
        let k = *self.shape().last().unwrap();
        assert_eq!(w.shape().len(), 2);
        assert_eq!(w.shape()[1], k);
        let n = w.shape()[0];
        let m = self.numel() / k;
        let mut out = vec![0.0; m * n];
        {
            let a = self.data();
            let b = w.data();
            mm_nt(&a[..], &b[..], m, k, n, &mut out);
        }
        let mut shape = self.shape().to_vec();
        *shape.last_mut().unwrap() = n;
        let (xc, wc) = (self.clone(), w.clone());
        Tensor::from_op(
            out,
            shape,
            vec![self.clone(), w.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let wd = wc.data();
                    let mut xg = xc.grad_mut();
                    mm_nn(go, &wd[..], m, n, k, &mut xg[..]);
                }
                {
                    let xd = xc.data();
                    let mut wg = wc.grad_mut();
                    mm_tn(go, &xd[..], m, n, k, &mut wg[..]);
                }
            }),
        )
    }

    pub fn dropout(&self, p: f32, rng: &mut Rng) -> Tensor {
        if p <= 0.0 {
            return self.clone();
        }
        let scale = 1.0 / (1.0 - p);
        let mask: Vec<f32> = (0..self.numel())
            .map(|_| if rng.uniform() < p { 0.0 } else { scale })
            .collect();
        let data: Vec<f32> = self.data().iter().zip(&mask).map(|(x, m)| x * m).collect();
        let ac = self.clone();
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let mut g = ac.grad_mut();
                for i in 0..go.len() {
                    g[i] += go[i] * mask[i];
                }
            }),
        )
    }

    pub fn bmm(&self, other: &Tensor) -> Tensor {
        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        assert_eq!(sa.len(), 3);
        assert_eq!(sb.len(), 3);
        assert_eq!(sa[0], sb[0]);
        assert_eq!(sa[2], sb[1]);
        let (bsz, m, k, n) = (sa[0], sa[1], sa[2], sb[2]);
        let mut out = vec![0.0; bsz * m * n];
        {
            let a = self.data();
            let b = other.data();
            for bi in 0..bsz {
                mm_nn(
                    &a[bi * m * k..][..m * k],
                    &b[bi * k * n..][..k * n],
                    m,
                    k,
                    n,
                    &mut out[bi * m * n..][..m * n],
                );
            }
        }
        let (ac, bc) = (self.clone(), other.clone());
        Tensor::from_op(
            out,
            vec![bsz, m, n],
            vec![self.clone(), other.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let bd = bc.data();
                    let mut ag = ac.grad_mut();
                    for bi in 0..bsz {
                        mm_nt(
                            &go[bi * m * n..][..m * n],
                            &bd[bi * k * n..][..k * n],
                            m,
                            n,
                            k,
                            &mut ag[bi * m * k..][..m * k],
                        );
                    }
                }
                {
                    let ad = ac.data();
                    let mut bg = bc.grad_mut();
                    for bi in 0..bsz {
                        mm_tn(
                            &ad[bi * m * k..][..m * k],
                            &go[bi * m * n..][..m * n],
                            m,
                            k,
                            n,
                            &mut bg[bi * k * n..][..k * n],
                        );
                    }
                }
            }),
        )
    }

    pub fn bmm_t(&self, other: &Tensor) -> Tensor {
        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        assert_eq!(sa.len(), 3);
        assert_eq!(sb.len(), 3);
        assert_eq!(sa[0], sb[0]);
        assert_eq!(sa[2], sb[2]);
        let (bsz, m, k, n) = (sa[0], sa[1], sa[2], sb[1]);
        let mut out = vec![0.0; bsz * m * n];
        {
            let a = self.data();
            let b = other.data();
            for bi in 0..bsz {
                mm_nt(
                    &a[bi * m * k..][..m * k],
                    &b[bi * n * k..][..n * k],
                    m,
                    k,
                    n,
                    &mut out[bi * m * n..][..m * n],
                );
            }
        }
        let (ac, bc) = (self.clone(), other.clone());
        Tensor::from_op(
            out,
            vec![bsz, m, n],
            vec![self.clone(), other.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let bd = bc.data();
                    let mut ag = ac.grad_mut();
                    for bi in 0..bsz {
                        mm_nn(
                            &go[bi * m * n..][..m * n],
                            &bd[bi * n * k..][..n * k],
                            m,
                            n,
                            k,
                            &mut ag[bi * m * k..][..m * k],
                        );
                    }
                }
                {
                    let ad = ac.data();
                    let mut bg = bc.grad_mut();
                    for bi in 0..bsz {
                        mm_tn(
                            &go[bi * m * n..][..m * n],
                            &ad[bi * m * k..][..m * k],
                            m,
                            n,
                            k,
                            &mut bg[bi * n * k..][..n * k],
                        );
                    }
                }
            }),
        )
    }

    pub fn causal_softmax(&self) -> Tensor {
        let dims = self.shape();
        let t = dims[dims.len() - 1];
        assert_eq!(dims[dims.len() - 2], t);
        let rows = self.numel() / t;
        let mut out = vec![0.0; self.numel()];
        {
            let d = self.data();
            for r in 0..rows {
                let i = r % t;
                let src = &d[r * t..][..t];
                let dst = &mut out[r * t..][..t];
                let mut mx = f32::NEG_INFINITY;
                for &v in &src[..=i] {
                    mx = mx.max(v);
                }
                let mut se = 0.0;
                for j in 0..=i {
                    let e = (src[j] - mx).exp();
                    dst[j] = e;
                    se += e;
                }
                let inv = 1.0 / se;
                for v in &mut dst[..=i] {
                    *v *= inv;
                }
            }
        }
        let ac = self.clone();
        Tensor::from_op(
            out,
            dims.to_vec(),
            vec![self.clone()],
            Box::new(move |yd: &[f32], go: &[f32]| {
                let mut g = ac.grad_mut();
                for r in 0..rows {
                    let i = r % t;
                    let y = &yd[r * t..][..t];
                    let gorow = &go[r * t..][..t];
                    let mut s = 0.0;
                    for j in 0..=i {
                        s += gorow[j] * y[j];
                    }
                    let grow = &mut g[r * t..][..t];
                    for j in 0..=i {
                        grow[j] += y[j] * (gorow[j] - s);
                    }
                }
            }),
        )
    }

    pub fn cross_entropy(&self, targets: &[i32]) -> Tensor {
        let v = *self.shape().last().unwrap();
        let rows = self.numel() / v;
        assert_eq!(targets.len(), rows);
        let mut probs = vec![0.0f32; rows * v];
        let mut count = 0usize;
        let mut loss = 0.0f64;
        {
            let d = self.data();
            for r in 0..rows {
                let src = &d[r * v..][..v];
                let p = &mut probs[r * v..][..v];
                let mx = src.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut se = 0.0f32;
                for j in 0..v {
                    let e = (src[j] - mx).exp();
                    p[j] = e;
                    se += e;
                }
                let inv = 1.0 / se;
                for x in p.iter_mut() {
                    *x *= inv;
                }
                let tgt = targets[r];
                if tgt >= 0 {
                    count += 1;

                    loss += -((src[tgt as usize] - mx) as f64 - (se as f64).ln());
                }
            }
        }
        let denom = count.max(1);
        let mean = (loss / denom as f64) as f32;
        let tv: Vec<i32> = targets.to_vec();
        let ac = self.clone();
        Tensor::from_op(
            vec![mean],
            vec![1],
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let g0 = go[0] / denom as f32;
                let mut g = ac.grad_mut();
                for r in 0..rows {
                    let tgt = tv[r];
                    if tgt < 0 {
                        continue;
                    }
                    let p = &probs[r * v..][..v];
                    let grow = &mut g[r * v..][..v];
                    for j in 0..v {
                        let onehot = if j == tgt as usize { 1.0 } else { 0.0 };
                        grow[j] += g0 * (p[j] - onehot);
                    }
                }
            }),
        )
    }

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor) -> Tensor {
        const EPS: f32 = 1e-5;
        let c = *self.shape().last().unwrap();
        assert_eq!(gamma.numel(), c);
        assert_eq!(beta.numel(), c);
        let rows = self.numel() / c;
        let mut out = vec![0.0; self.numel()];
        let mut xhat = vec![0.0f32; self.numel()];
        let mut rstd = vec![0.0f32; rows];
        {
            let d = self.data();
            let gm = gamma.data();
            let bt = beta.data();
            for r in 0..rows {
                let src = &d[r * c..][..c];
                let mu = src.iter().sum::<f32>() / c as f32;
                let var = src.iter().map(|&x| (x - mu) * (x - mu)).sum::<f32>() / c as f32;
                let rs = 1.0 / (var + EPS).sqrt();
                rstd[r] = rs;
                for j in 0..c {
                    let xh = (src[j] - mu) * rs;
                    xhat[r * c + j] = xh;
                    out[r * c + j] = gm[j] * xh + bt[j];
                }
            }
        }
        let (xc, gc, bc) = (self.clone(), gamma.clone(), beta.clone());
        Tensor::from_op(
            out,
            self.shape().to_vec(),
            vec![self.clone(), gamma.clone(), beta.clone()],
            Box::new(move |_, go: &[f32]| {
                {
                    let mut gg = gc.grad_mut();
                    for r in 0..rows {
                        for j in 0..c {
                            gg[j] += go[r * c + j] * xhat[r * c + j];
                        }
                    }
                }
                {
                    let mut bg = bc.grad_mut();
                    for r in 0..rows {
                        for j in 0..c {
                            bg[j] += go[r * c + j];
                        }
                    }
                }
                {
                    let gm = gc.data();
                    let mut xg = xc.grad_mut();
                    for r in 0..rows {
                        let mut mean_g = 0.0f32;
                        let mut mean_gx = 0.0f32;
                        for j in 0..c {
                            let ghat = go[r * c + j] * gm[j];
                            mean_g += ghat;
                            mean_gx += ghat * xhat[r * c + j];
                        }
                        mean_g /= c as f32;
                        mean_gx /= c as f32;
                        let rs = rstd[r];
                        for j in 0..c {
                            let ghat = go[r * c + j] * gm[j];
                            xg[r * c + j] += rs * (ghat - mean_g - xhat[r * c + j] * mean_gx);
                        }
                    }
                }
            }),
        )
    }

    pub fn gather_rows(&self, idx: &[u32], prefix_shape: &[usize]) -> Tensor {
        assert_eq!(self.shape().len(), 2);
        let c = self.shape()[1];
        assert_eq!(idx.len(), prefix_shape.iter().product::<usize>());
        let mut out = vec![0.0; idx.len() * c];
        {
            let d = self.data();
            for (i, &id) in idx.iter().enumerate() {
                out[i * c..][..c].copy_from_slice(&d[id as usize * c..][..c]);
            }
        }
        let mut shape = prefix_shape.to_vec();
        shape.push(c);
        let wc = self.clone();
        let idxv: Vec<u32> = idx.to_vec();
        Tensor::from_op(
            out,
            shape,
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let mut g = wc.grad_mut();
                for (i, &id) in idxv.iter().enumerate() {
                    let dst = &mut g[id as usize * c..][..c];
                    for (dj, &oj) in dst.iter_mut().zip(&go[i * c..][..c]) {
                        *dj += oj;
                    }
                }
            }),
        )
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        assert_eq!(self.numel(), new_shape.iter().product::<usize>());
        let data = self.data().clone();
        let ac = self.clone();
        Tensor::from_op(
            data,
            new_shape.to_vec(),
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let mut g = ac.grad_mut();
                for (gi, &o) in g.iter_mut().zip(go) {
                    *gi += o;
                }
            }),
        )
    }

    pub fn cat(parts: &[Tensor]) -> Tensor {
        assert!(!parts.is_empty());
        let lead = &parts[0].shape()[..parts[0].shape().len() - 1];
        let rows: usize = lead.iter().product();
        let widths: Vec<usize> = parts
            .iter()
            .map(|p| {
                assert_eq!(&p.shape()[..p.shape().len() - 1], lead);
                *p.shape().last().unwrap()
            })
            .collect();
        let total: usize = widths.iter().sum();
        let mut out = vec![0.0; rows * total];
        {
            let datas: Vec<Ref<Vec<f32>>> = parts.iter().map(|p| p.data()).collect();
            for r in 0..rows {
                let mut off = 0;
                for (pi, w) in widths.iter().enumerate() {
                    out[r * total + off..][..*w].copy_from_slice(&datas[pi][r * w..][..*w]);
                    off += w;
                }
            }
        }
        let mut shape = lead.to_vec();
        shape.push(total);
        let clones: Vec<Tensor> = parts.to_vec();
        let widths_c = widths.clone();
        Tensor::from_op(
            out,
            shape,
            parts.to_vec(),
            Box::new(move |_, go: &[f32]| {
                let mut off = 0;
                for (pi, w) in widths_c.iter().enumerate() {
                    let mut g = clones[pi].grad_mut();
                    for r in 0..rows {
                        for j in 0..*w {
                            g[r * w + j] += go[r * total + off + j];
                        }
                    }
                    off += w;
                }
            }),
        )
    }

    pub fn slice_last(&self, start: usize, len: usize) -> Tensor {
        let w = *self.shape().last().unwrap();
        assert!(start + len <= w);
        let rows = self.numel() / w;
        let mut out = vec![0.0; rows * len];
        {
            let d = self.data();
            for r in 0..rows {
                out[r * len..][..len].copy_from_slice(&d[r * w + start..][..len]);
            }
        }
        let mut shape = self.shape().to_vec();
        *shape.last_mut().unwrap() = len;
        let ac = self.clone();
        Tensor::from_op(
            out,
            shape,
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let mut g = ac.grad_mut();
                for r in 0..rows {
                    for j in 0..len {
                        g[r * w + start + j] += go[r * len + j];
                    }
                }
            }),
        )
    }

    pub fn transpose12(&self) -> Tensor {
        let s = self.shape();
        assert_eq!(s.len(), 4);
        let (a, b, c, d) = (s[0], s[1], s[2], s[3]);
        let mut out = vec![0.0; self.numel()];
        {
            let src = self.data();
            for ai in 0..a {
                for bi in 0..b {
                    for ci in 0..c {
                        let from = ((ai * b + bi) * c + ci) * d;
                        let to = ((ai * c + ci) * b + bi) * d;
                        out[to..to + d].copy_from_slice(&src[from..from + d]);
                    }
                }
            }
        }
        let ac = self.clone();
        Tensor::from_op(
            out,
            vec![a, c, b, d],
            vec![self.clone()],
            Box::new(move |_, go: &[f32]| {
                let mut g = ac.grad_mut();
                for ai in 0..a {
                    for bi in 0..b {
                        for ci in 0..c {
                            let from = ((ai * b + bi) * c + ci) * d;
                            let to = ((ai * c + ci) * b + bi) * d;
                            for j in 0..d {
                                g[from + j] += go[to + j];
                            }
                        }
                    }
                }
            }),
        )
    }

    pub fn stack_rows(parts: &[Tensor]) -> Tensor {
        assert!(!parts.is_empty());
        let t = parts.len();
        let (b, h) = (parts[0].shape()[0], parts[0].shape()[1]);
        let mut out = vec![0.0; b * t * h];
        {
            for (ti, p) in parts.iter().enumerate() {
                assert_eq!(p.shape(), &[b, h]);
                let d = p.data();
                for bi in 0..b {
                    out[(bi * t + ti) * h..][..h].copy_from_slice(&d[bi * h..][..h]);
                }
            }
        }
        let clones: Vec<Tensor> = parts.to_vec();
        Tensor::from_op(
            out,
            vec![b, t, h],
            parts.to_vec(),
            Box::new(move |_, go: &[f32]| {
                for (ti, p) in clones.iter().enumerate() {
                    let mut g = p.grad_mut();
                    for bi in 0..b {
                        for j in 0..h {
                            g[bi * h + j] += go[(bi * t + ti) * h + j];
                        }
                    }
                }
            }),
        )
    }
}

fn n_threads() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MAKEMORE_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
            .clamp(1, 16)
    })
}

const PAR_MIN_WORK: usize = 1 << 21;

fn par_row_chunks<F>(out: &mut [f32], row_len: usize, work: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let m = out.len() / row_len.max(1);
    let nt = n_threads();
    if work < PAR_MIN_WORK || nt <= 1 || m < 2 {
        f(0, out);
        return;
    }
    let nt = nt.min(m);
    let chunk = (m + nt - 1) / nt;
    std::thread::scope(|s| {
        for (ci, oc) in out.chunks_mut(chunk * row_len).enumerate() {
            let f = &f;
            s.spawn(move || f(ci * chunk, oc));
        }
    });
}

pub(crate) fn mm_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    par_row_chunks(out, n, 2 * m * k * n, |i0, oc| {
        for (ii, orow) in oc.chunks_mut(n).enumerate() {
            let arow = &a[(i0 + ii) * k..][..k];
            for (kk, &av) in arow.iter().enumerate() {
                let brow = &b[kk * n..][..n];
                for (o, &bv) in orow.iter_mut().zip(brow) {
                    *o += av * bv;
                }
            }
        }
    });
}

pub(crate) fn mm_nt(a: &[f32], b: &[f32], m: usize, n: usize, kdim: usize, out: &mut [f32]) {
    par_row_chunks(out, kdim, 2 * m * n * kdim, |i0, oc| {
        for (ii, orow) in oc.chunks_mut(kdim).enumerate() {
            let arow = &a[(i0 + ii) * n..][..n];
            for (kk, o) in orow.iter_mut().enumerate() {
                let brow = &b[kk * n..][..n];
                let mut s = 0.0;
                for (x, y) in arow.iter().zip(brow) {
                    s += x * y;
                }
                *o += s;
            }
        }
    });
}

pub(crate) fn mm_tn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    par_row_chunks(out, n, 2 * m * k * n, |k0, oc| {
        let krows = oc.len() / n;
        for mi in 0..m {
            let brow = &b[mi * n..][..n];
            let abase = mi * k + k0;
            for kk in 0..krows {
                let av = a[abase + kk];
                let orow = &mut oc[kk * n..][..n];
                for (o, &bv) in orow.iter_mut().zip(brow) {
                    *o += av * bv;
                }
            }
        }
    });
}
