use std::collections::HashMap;
use std::io::{self, Read};

use crate::json::{parse, Json};

pub struct SafeTensors {
    pub tensors: HashMap<String, (Vec<usize>, usize, usize)>,
    buf: Vec<u8>,
}

impl SafeTensors {
    pub fn load(path: &str) -> io::Result<SafeTensors> {
        let mut f = std::fs::File::open(path)?;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)?;
        let hlen = u64::from_le_bytes(len8) as usize;
        let mut hbuf = vec![0u8; hlen];
        f.read_exact(&mut hbuf)?;
        let header = String::from_utf8(hbuf).map_err(bad)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;

        let j = parse(&header).map_err(bad)?;
        let obj = j.as_obj().ok_or_else(|| bad("header is not an object"))?;
        let mut tensors = HashMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta
                .get("dtype")
                .and_then(Json::as_str)
                .ok_or_else(|| bad("missing dtype"))?;
            if dtype != "F32" {
                return Err(bad(format!(
                    "{name}: dtype {dtype} unsupported (convert to float32)"
                )));
            }
            let shape: Vec<usize> = meta
                .get("shape")
                .and_then(Json::as_arr)
                .ok_or_else(|| bad("missing shape"))?
                .iter()
                .map(|v| v.as_usize().unwrap())
                .collect();
            let off = meta
                .get("data_offsets")
                .and_then(Json::as_arr)
                .ok_or_else(|| bad("missing offsets"))?;
            let (s, e) = (off[0].as_usize().unwrap(), off[1].as_usize().unwrap());
            if e > buf.len() || s > e {
                return Err(bad(format!("{name}: offsets out of range")));
            }
            tensors.insert(name.clone(), (shape, s, e));
        }
        Ok(SafeTensors { tensors, buf })
    }

    pub fn get(&self, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let (shape, s, e) = self.tensors.get(name)?;
        let data: Vec<f32> = self.buf[*s..*e]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Some((shape.clone(), data))
    }

    pub fn names(&self) -> Vec<&String> {
        let mut v: Vec<&String> = self.tensors.keys().collect();
        v.sort();
        v
    }
}

fn bad(e: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}
