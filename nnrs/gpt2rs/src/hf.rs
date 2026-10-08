use std::io;

use crate::json::{parse, Json};
use crate::model::{Gpt, GptConfig};
use crate::rng::Rng;
use crate::safetensors::SafeTensors;

pub fn load_config(path: &str) -> io::Result<GptConfig> {
    let text = std::fs::read_to_string(path)?;
    let j = parse(&text).map_err(bad)?;
    let need = |k: &str| -> io::Result<usize> {
        j.get(k)
            .and_then(Json::as_usize)
            .ok_or_else(|| bad(format!("config missing {k}")))
    };
    if let Some(act) = j.get("activation_function").and_then(Json::as_str) {
        if act != "gelu_new" {
            return Err(bad(format!(
                "activation {act} unsupported (this is a GPT-2 loader)"
            )));
        }
    }
    Ok(GptConfig {
        vocab_size: need("vocab_size")?,
        block_size: need("n_positions")?,
        n_layer: need("n_layer")?,
        n_head: need("n_head")?,
        n_embd: need("n_embd")?,
        dropout: 0.0,
    })
}

pub fn load_gpt(cfg: &GptConfig, weights_path: &str) -> io::Result<Gpt> {
    let st = SafeTensors::load(weights_path)?;
    let mut rng = Rng::new(0);
    let model = Gpt::new(cfg, &mut rng);
    for (name, p) in model.params() {
        let (shape, data) = st
            .get(&name)
            .or_else(|| st.get(&format!("transformer.{name}")))
            .ok_or_else(|| bad(format!("weights missing tensor {name}")))?;
        if shape != p.shape() {
            return Err(bad(format!(
                "{name}: shape {shape:?} != expected {:?}",
                p.shape()
            )));
        }
        p.data_mut().copy_from_slice(&data);
    }
    Ok(model)
}

fn bad(e: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}
