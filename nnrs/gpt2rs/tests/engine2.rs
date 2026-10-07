use gpt2rs::rng::Rng;
use gpt2rs::tensor::{self, Tensor};

const EPS: f32 = 1e-2;

fn check_grads(inputs: &[&Tensor], build: &dyn Fn() -> Tensor) {
    for t in inputs {
        t.zero_grad();
    }
    let loss = build();
    loss.backward();
    let analytic: Vec<Vec<f32>> = inputs.iter().map(|t| t.grad().clone()).collect();
    for (ti, t) in inputs.iter().enumerate() {
        for j in 0..t.numel() {
            let orig = t.data()[j];
            t.data_mut()[j] = orig + EPS;
            let lp = build().item();
            t.data_mut()[j] = orig - EPS;
            let lm = build().item();
            t.data_mut()[j] = orig;
            let numeric = (lp - lm) / (2.0 * EPS);
            let a = analytic[ti][j];
            let tol = 1e-2 + 3e-2 * numeric.abs().max(a.abs());
            assert!(
                (a - numeric).abs() <= tol,
                "input {ti} elem {j}: analytic {a} vs numeric {numeric}"
            );
        }
    }
}

#[test]
fn matmul_t_gradcheck() {
    let mut rng = Rng::new(1);
    let x = Tensor::randn(&mut rng, &[3, 4], 1.0);
    let w = Tensor::randn(&mut rng, &[5, 4], 1.0);
    let r = Tensor::randn(&mut rng, &[3, 5], 1.0);
    check_grads(&[&x, &w], &|| x.matmul_t(&w).mul(&r).sum());

    let wt = {
        let d = w.data();
        let mut t = vec![0.0; 20];
        for i in 0..5 {
            for j in 0..4 {
                t[j * 5 + i] = d[i * 4 + j];
            }
        }
        Tensor::leaf(t, &[4, 5])
    };
    let a = x.matmul_t(&w);
    let b = x.matmul(&wt);
    for (u, v) in a.data().iter().zip(b.data().iter()) {
        assert!((u - v).abs() < 1e-5);
    }
}

#[test]
fn dropout_gradcheck_and_stats() {
    let mut rng = Rng::new(2);
    let x = Tensor::randn(&mut rng, &[6, 5], 1.0);
    let r = Tensor::randn(&mut rng, &[6, 5], 1.0);

    check_grads(&[&x], &|| {
        let mut drop_rng = Rng::new(99);
        x.dropout(0.4, &mut drop_rng).mul(&r).sum()
    });

    let y = x.dropout(0.0, &mut rng);
    assert!(std::ptr::eq(y.data().as_ptr(), x.data().as_ptr()));

    let big = Tensor::ones(&[10_000]);
    let mut m = 0.0f64;
    let mut rng2 = Rng::new(3);
    let d = big.dropout(0.3, &mut rng2);
    for &v in d.data().iter() {
        m += v as f64;
    }
    let mean = m / 10_000.0;
    assert!((mean - 1.0).abs() < 0.05, "inverted dropout mean {mean}");
}

#[test]
fn no_grad_mode_skips_graph() {
    let mut rng = Rng::new(4);
    let x = Tensor::randn(&mut rng, &[4, 4], 1.0);
    tensor::set_no_grad(true);
    let y = x.tanh().sum();
    tensor::set_no_grad(false);

    x.zero_grad();
    y.backward();
    assert!(
        x.grad().iter().all(|&g| g == 0.0),
        "no_grad leaked gradients"
    );
}
