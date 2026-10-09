use std::fs::File;
use std::io::{self, Read, Write};

use crate::model::GptConfig;
use crate::tensor::Tensor;

const MAGIC: &[u8] = b"GPT2-RS-v1\n";

pub struct Meta {
    pub tokenizer: String,
    pub cfg: GptConfig,
    pub best_loss: f32,
    pub seed: u64,
}

pub fn save(path: &str, meta: &Meta, params: &[(String, Tensor)]) -> io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(MAGIC)?;
    write_str(&mut f, &meta.tokenizer)?;
    f.write_all(&meta.seed.to_le_bytes())?;
    f.write_all(&meta.best_loss.to_le_bytes())?;
    for v in [
        meta.cfg.vocab_size,
        meta.cfg.block_size,
        meta.cfg.n_layer,
        meta.cfg.n_head,
        meta.cfg.n_embd,
    ] {
        f.write_all(&(v as u32).to_le_bytes())?;
    }
    f.write_all(&meta.cfg.dropout.to_le_bytes())?;
    f.write_all(&(params.len() as u32).to_le_bytes())?;
    for (name, p) in params {
        write_str(&mut f, name)?;
        f.write_all(&(p.shape().len() as u32).to_le_bytes())?;
        for &d in p.shape() {
            f.write_all(&(d as u32).to_le_bytes())?;
        }
        let data = p.data();
        let mut buf = Vec::with_capacity(data.len() * 4);
        for &x in data.iter() {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    Ok(())
}

pub fn load(path: &str) -> io::Result<(Meta, Vec<(String, Vec<usize>, Vec<f32>)>)> {
    let mut f = File::open(path)?;
    let mut magic = vec![0u8; MAGIC.len()];
    f.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a gpt2-rust checkpoint",
        ));
    }
    let tokenizer = read_str(&mut f)?;
    let seed = u64::from_le_bytes(read_arr(&mut f)?);
    let best_loss = f32::from_le_bytes(read_arr(&mut f)?);
    let mut c = [0usize; 5];
    for v in c.iter_mut() {
        *v = u32::from_le_bytes(read_arr(&mut f)?) as usize;
    }
    let dropout = f32::from_le_bytes(read_arr(&mut f)?);
    let cfg = GptConfig {
        vocab_size: c[0],
        block_size: c[1],
        n_layer: c[2],
        n_head: c[3],
        n_embd: c[4],
        dropout,
    };
    let n = u32::from_le_bytes(read_arr(&mut f)?) as usize;
    let mut params = Vec::with_capacity(n);
    for _ in 0..n {
        let name = read_str(&mut f)?;
        let ndims = u32::from_le_bytes(read_arr(&mut f)?) as usize;
        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            dims.push(u32::from_le_bytes(read_arr(&mut f)?) as usize);
        }
        let numel: usize = dims.iter().product();
        let mut buf = vec![0u8; numel * 4];
        f.read_exact(&mut buf)?;
        let data = buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        params.push((name, dims, data));
    }
    Ok((
        Meta {
            tokenizer,
            cfg,
            best_loss,
            seed,
        },
        params,
    ))
}

pub fn restore(model_params: &[(String, Tensor)], loaded: &[(String, Vec<usize>, Vec<f32>)]) {
    assert_eq!(model_params.len(), loaded.len(), "parameter count mismatch");
    for ((name, p), (lname, ldims, ldata)) in model_params.iter().zip(loaded) {
        assert_eq!(name, lname, "parameter name mismatch: {name} vs {lname}");
        assert_eq!(p.shape(), &ldims[..], "shape mismatch for {name}");
        p.data_mut().copy_from_slice(ldata);
    }
}

fn write_str(f: &mut File, s: &str) -> io::Result<()> {
    f.write_all(&(s.len() as u32).to_le_bytes())?;
    f.write_all(s.as_bytes())
}

fn read_str(f: &mut File) -> io::Result<String> {
    let len = u32::from_le_bytes(read_arr(f)?) as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_arr<const N: usize>(f: &mut File) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    f.read_exact(&mut buf)?;
    Ok(buf)
}
