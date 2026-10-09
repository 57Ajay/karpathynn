use std::io::Write as _;
use std::time::Instant;

use gpt2rs::bpe_gpt2::Gpt2Tokenizer;
use gpt2rs::checkpoint::{self, Meta};
use gpt2rs::data::{CharTok, Dataset, Tok};
use gpt2rs::infer;
use gpt2rs::model::{Gpt, GptConfig};
use gpt2rs::optim::AdamW;
use gpt2rs::rng::Rng;
use gpt2rs::safetensors::SafeTensors;
use gpt2rs::tensor;

const USAGE: &str = "gpt2-rust: GPT-2 from scratch in pure Rust

USAGE:
  gpt2 train    --data <file> [--steps N] [--out dir] [--batch-size N] [--block-size N]
                [--n-layer N] [--n-head N] [--n-embd N] [--dropout P]
                [--learning-rate F] [--eval-every N] [--seed N]
                (character-level tokenizer, video-7 style)
  gpt2 generate --checkpoint <dir/model.bin> [--prompt S] [--tokens N]
                [--temperature F] [--top-k N] [--seed N]
  gpt2 generate --hf-config <config.json> --hf-weights <model.safetensors>
                [--prompt S] [--tokens N] [--temperature F] [--top-k N]
                (runs real GPT-2 checkpoints with the BPE tokenizer; see
                 scripts/fetch_gpt2_weights.py to download the 124M files)
  gpt2 verify   --hf-dir <dir>      compare our forward pass to a HuggingFace
                                    reference dump (config.json, model.safetensors,
                                    reference.json) — used by tests/hf_tiny
  gpt2 info     --hf-weights <f>    list tensors in a safetensors file
";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| {
        args.get(i + 1)
            .unwrap_or_else(|| panic!("missing value for {name}"))
            .clone()
    })
}

fn flag_or<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    flag(args, name)
        .map(|s| s.parse().expect("bad flag value"))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str).unwrap_or("") {
        "train" => train(&args),
        "generate" => generate(&args),
        "verify" => verify(&args),
        "info" => {
            let st =
                SafeTensors::load(&flag(&args, "--hf-weights").expect("--hf-weights required"))
                    .unwrap();
            for name in st.names() {
                let (shape, s, e) = &st.tensors[name];
                println!("{name}  {shape:?}  {} bytes", e - s);
            }
        }
        "--help" | "-h" | "help" => print!("{USAGE}"),
        other => {
            eprintln!("unknown command {other:?}\n{USAGE}");
            std::process::exit(1);
        }
    }
}

fn train(args: &[String]) {
    let data_path = flag(args, "--data").expect("--data required");
    let out_dir = flag(args, "--out").unwrap_or_else(|| "out".into());
    let steps: u64 = flag_or(args, "--steps", 2000);
    let batch: usize = flag_or(args, "--batch-size", 16);
    let seed: u64 = flag_or(args, "--seed", 1337);
    let lr: f32 = flag_or(args, "--learning-rate", 1e-3);
    let eval_every: u64 = flag_or(args, "--eval-every", 250);

    let text = std::fs::read_to_string(&data_path).expect("cannot read data");
    let tok = CharTok::from_text(&text);
    let cfg = GptConfig {
        vocab_size: tok.vocab_size(),
        block_size: flag_or(args, "--block-size", 64),
        n_layer: flag_or(args, "--n-layer", 4),
        n_head: flag_or(args, "--n-head", 4),
        n_embd: flag_or(args, "--n-embd", 128),
        dropout: flag_or(args, "--dropout", 0.0),
    };
    println!("data: {} chars, vocab {}", text.len(), tok.vocab_size());

    let ds = Dataset::new(tok.encode(&text));
    let (train_ids, val_ids) = ds.split();
    let mut rng = Rng::new(seed);
    let mut init_rng = Rng::new(seed ^ 0x67707432);
    let model = Gpt::new(&cfg, &mut init_rng);
    println!("model: {:?}", cfg);
    println!("#params: {}", model.num_params());

    let params = model.params();
    let mut opt = AdamW::new(params.iter().map(|(_, p)| p.clone()).collect(), lr, 0.1);
    std::fs::create_dir_all(&out_dir).ok();
    let model_path = format!("{out_dir}/model.bin");
    let mut best = f32::INFINITY;

    for step in 0..steps {
        let t0 = Instant::now();
        let (x, y) = Dataset::sample_batch(train_ids, &mut rng, batch, cfg.block_size);
        let loss = model.loss(&x, &y, batch, cfg.block_size, cfg.dropout, &mut rng);
        assert!(loss.item().is_finite(), "loss diverged at step {step}");
        opt.zero_grad();
        loss.backward();
        opt.step();
        if step % 10 == 0 {
            println!(
                "step {step} | loss {:.4} | {:.0}ms",
                loss.item(),
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        if step > 0 && (step % eval_every == 0 || step == steps - 1) {
            let val = evaluate(&model, val_ids, &mut rng, batch, cfg.block_size);
            println!("step {step} | val loss {val:.4}");
            if val < best {
                best = val;
                let meta = Meta {
                    tokenizer: format!("char:{}", tok.chars.iter().collect::<String>()),
                    cfg,
                    best_loss: best,
                    seed,
                };
                checkpoint::save(&model_path, &meta, &params).expect("save failed");
                println!("saved {model_path} (val {val:.4})");
            }

            let sample = sample_text(
                &model,
                &Tok::Char(CharTok::from_chars(tok.chars.clone())),
                "\n",
                150,
                &mut rng,
                0.9,
                None,
            );
            println!("--- sample ---\n{sample}\n--------------");
        }
    }
}

fn evaluate(model: &Gpt, val: &[u32], rng: &mut Rng, batch: usize, block: usize) -> f32 {
    tensor::set_no_grad(true);
    let mut tot = 0.0;
    let n = 8;
    for _ in 0..n {
        let (x, y) = Dataset::sample_batch(val, rng, batch, block);
        tot += model.loss(&x, &y, batch, block, 0.0, rng).item();
    }
    tensor::set_no_grad(false);
    tot / n as f32
}

fn sample_text(
    model: &Gpt,
    tok: &Tok,
    prompt: &str,
    n_tokens: usize,
    rng: &mut Rng,
    temperature: f32,
    top_k: Option<usize>,
) -> String {
    let prompt_ids = tok.encode(prompt);
    let out = infer::generate(
        model,
        &prompt_ids,
        n_tokens,
        rng,
        temperature,
        top_k,
        |_| {},
    );
    tok.decode(&out)
}

fn load_checkpoint(path: &str) -> (Gpt, Tok) {
    let (meta, params) = checkpoint::load(path).expect("cannot load checkpoint");
    let mut rng = Rng::new(0);
    let model = Gpt::new(&meta.cfg, &mut rng);
    checkpoint::restore(&model.params(), &params);
    let tok = if let Some(chars) = meta.tokenizer.strip_prefix("char:") {
        Tok::Char(CharTok::from_chars(chars.chars().collect()))
    } else {
        Tok::Bpe(Box::new(
            Gpt2Tokenizer::load("gpt2/vocab.bpe").expect("gpt2/vocab.bpe missing"),
        ))
    };
    (model, tok)
}

fn generate(args: &[String]) {
    let tokens: usize = flag_or(args, "--tokens", 200);
    let temperature: f32 = flag_or(args, "--temperature", 0.9);
    let top_k = flag(args, "--top-k").map(|s| s.parse().unwrap());
    let mut rng = Rng::new(flag_or(args, "--seed", 42u64));

    let (model, tok) = if let Some(w) = flag(args, "--hf-weights") {
        let cfg = gpt2rs::hf::load_config(
            &flag(args, "--hf-config").expect("--hf-config required with --hf-weights"),
        )
        .unwrap();
        println!("loading {w} ...");
        let t0 = Instant::now();
        let model = gpt2rs::hf::load_gpt(&cfg, &w).unwrap();
        println!(
            "loaded {} params in {:.1}s",
            model.num_params(),
            t0.elapsed().as_secs_f64()
        );
        (
            model,
            Tok::Bpe(Box::new(
                Gpt2Tokenizer::load("gpt2/vocab.bpe").expect("gpt2/vocab.bpe missing"),
            )),
        )
    } else {
        load_checkpoint(&flag(args, "--checkpoint").expect("--checkpoint or --hf-weights required"))
    };

    let prompt = flag(args, "--prompt").unwrap_or_else(|| "\n".into());
    let prompt_ids = tok.encode(&prompt);
    assert!(!prompt_ids.is_empty(), "prompt tokenized to nothing");
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let mut count = 0usize;
    infer::generate(
        &model,
        &prompt_ids,
        tokens,
        &mut rng,
        temperature,
        top_k,
        |t| {
            let s = tok.decode(&[t]);
            print!("{s}");
            std::io::stdout().flush().ok();
            count += 1;
        },
    );
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "\n({count} tokens in {dt:.1}s, {:.0} ms/token)",
        dt * 1000.0 / count.max(1) as f64
    );
}

fn verify(args: &[String]) {
    let dir = flag(args, "--hf-dir").expect("--hf-dir required");
    let cfg = gpt2rs::hf::load_config(&format!("{dir}/config.json")).unwrap();
    let model = gpt2rs::hf::load_gpt(&cfg, &format!("{dir}/model.safetensors")).unwrap();
    let reference =
        std::fs::read_to_string(format!("{dir}/reference.json")).expect("reference.json missing");
    let j = gpt2rs::json::parse(&reference).unwrap();
    let ids: Vec<u32> = j
        .get("input_ids")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|v| v.as_usize().unwrap() as u32)
        .collect();
    let expect: Vec<f32> = j
        .get("logits_all")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    tensor::set_no_grad(true);
    let mut rng = Rng::new(0);
    let logits = model.forward(&ids, 1, ids.len(), 0.0, &mut rng);
    tensor::set_no_grad(false);
    let got = logits.data();
    assert_eq!(got.len(), expect.len(), "logit count mismatch");
    let mut max_diff = 0.0f32;
    for i in 0..expect.len() {
        max_diff = max_diff.max((got[i] - expect[i]).abs());
    }
    println!(
        "compared {} logits against HuggingFace: max abs diff = {max_diff:.2e}",
        expect.len()
    );
    assert!(max_diff < 2e-3, "PARITY FAILED");
    println!("PARITY OK — this implementation matches transformers' GPT2LMHeadModel");
}
