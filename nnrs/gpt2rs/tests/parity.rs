use gpt2rs::hf;
use gpt2rs::json;
use gpt2rs::rng::Rng;
use gpt2rs::tensor;

#[test]
fn forward_matches_huggingface() {
    let dir = "tests/hf_tiny";
    let cfg = hf::load_config(&format!("{dir}/config.json")).unwrap();
    let model = hf::load_gpt(&cfg, &format!("{dir}/model.safetensors")).unwrap();

    let reference = std::fs::read_to_string(format!("{dir}/reference.json")).unwrap();
    let j = json::parse(&reference).unwrap();
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
    assert_eq!(got.len(), expect.len());
    let mut max_diff = 0.0f32;
    for i in 0..expect.len() {
        max_diff = max_diff.max((got[i] - expect[i]).abs());
    }
    println!("max abs logit diff vs transformers: {max_diff:.2e}");
    assert!(
        max_diff < 2e-3,
        "diverged from HuggingFace: max diff {max_diff}"
    );
}

#[test]
fn tied_head_means_no_lm_head_tensor() {
    let st = gpt2rs::safetensors::SafeTensors::load("tests/hf_tiny/model.safetensors").unwrap();
    assert!(st.get("lm_head.weight").is_none());
    assert!(st.get("transformer.wte.weight").is_some() || st.get("wte.weight").is_some());
}
