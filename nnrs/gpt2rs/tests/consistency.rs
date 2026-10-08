use gpt2rs::infer::{self, KvCache};
use gpt2rs::model::{Gpt, GptConfig};
use gpt2rs::rng::Rng;
use gpt2rs::tensor;

#[test]
fn kv_cache_matches_full_forward() {
    let cfg = GptConfig {
        vocab_size: 91,
        block_size: 16,
        n_layer: 3,
        n_head: 2,
        n_embd: 48,
        dropout: 0.0,
    };
    let mut rng = Rng::new(7);
    let model = Gpt::new(&cfg, &mut rng);
    let ids: Vec<u32> = (0..12).map(|_| rng.below(cfg.vocab_size) as u32).collect();

    tensor::set_no_grad(true);
    let full = model.forward(&ids, 1, ids.len(), 0.0, &mut rng);
    tensor::set_no_grad(false);
    let fd = full.data();

    let mut cache = KvCache::new(&model);
    let mut max_diff = 0.0f32;
    for (t, &tok) in ids.iter().enumerate() {
        let logits = infer::step(&model, &mut cache, tok, true).unwrap();
        for v in 0..cfg.vocab_size {
            let diff = (logits[v] - fd[t * cfg.vocab_size + v]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    assert!(
        max_diff < 1e-4,
        "KV-cache path diverged from full forward: {max_diff}"
    );
}

#[test]
fn generation_slides_past_block_size() {
    let cfg = GptConfig {
        vocab_size: 40,
        block_size: 16,
        n_layer: 2,
        n_head: 2,
        n_embd: 32,
        dropout: 0.0,
    };
    let mut rng = Rng::new(21);
    let model = Gpt::new(&cfg, &mut rng);
    let prompt = [1u32, 2, 3];
    let mut r = Rng::new(5);
    let out = infer::generate(&model, &prompt, 50, &mut r, 1.0, None, |_| {});
    assert_eq!(
        out.len(),
        53,
        "generation must continue past block_size via the sliding window"
    );
}

#[test]
fn generation_is_deterministic_given_seed() {
    let cfg = GptConfig {
        vocab_size: 50,
        block_size: 32,
        n_layer: 2,
        n_head: 2,
        n_embd: 32,
        dropout: 0.0,
    };
    let mut rng = Rng::new(11);
    let model = Gpt::new(&cfg, &mut rng);
    let prompt = [3u32, 14, 15];
    let mut r1 = Rng::new(123);
    let mut r2 = Rng::new(123);
    let a = infer::generate(&model, &prompt, 20, &mut r1, 1.0, Some(10), |_| {});
    let b = infer::generate(&model, &prompt, 20, &mut r2, 1.0, Some(10), |_| {});
    assert_eq!(a, b);
    assert_eq!(&a[..3], &prompt);
    assert_eq!(a.len(), 23);
}
