use std::collections::HashMap;
use std::io::{self, Read};

use crate::split::{split, SplitPattern};
use crate::tokenizer::{encode_ids, Pair};

pub const ENDOFTEXT: &str = "<|endoftext|>";
pub const ENDOFTEXT_ID: u32 = 50256;

pub struct Gpt2Tokenizer {
    merges: HashMap<Pair, u32>,
    vocab: HashMap<u32, Vec<u8>>,
    byte_to_id: [u32; 256],
}

fn bytes_to_unicode() -> [char; 256] {
    let mut out = ['\0'; 256];
    let printable = (33u32..=126).chain(161..=172).chain(174..=255);
    let keep: Vec<u32> = printable.collect();
    let mut n = 0;
    for b in 0u32..256 {
        if keep.contains(&b) {
            out[b as usize] = char::from_u32(b).unwrap();
        } else {
            out[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    out
}

impl Gpt2Tokenizer {
    pub fn load(vocab_bpe_path: &str) -> io::Result<Gpt2Tokenizer> {
        let mut text = String::new();
        std::fs::File::open(vocab_bpe_path)?.read_to_string(&mut text)?;
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);

        let b2u = bytes_to_unicode();
        let mut sorted: Vec<(char, u8)> = b2u
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        sorted.sort_by_key(|&(c, _)| c as u32);
        let mut byte_to_id = [0u32; 256];
        let mut str_to_id: HashMap<String, u32> = HashMap::new();
        let mut vocab: HashMap<u32, Vec<u8>> = HashMap::new();
        for (id, &(c, b)) in sorted.iter().enumerate() {
            byte_to_id[b as usize] = id as u32;
            str_to_id.insert(c.to_string(), id as u32);
            vocab.insert(id as u32, vec![b]);
        }

        let mut merges = HashMap::new();
        let mut rank = 0u32;
        for line in text.lines() {
            if line.starts_with("#version") || line.is_empty() {
                continue;
            }
            let (a, b) = line
                .split_once(' ')
                .ok_or_else(|| bad(format!("bad merge line: {line:?}")))?;
            let ida = *str_to_id
                .get(a)
                .ok_or_else(|| bad(format!("unknown merge part: {a:?}")))?;
            let idb = *str_to_id
                .get(b)
                .ok_or_else(|| bad(format!("unknown merge part: {b:?}")))?;
            let id = 256 + rank;
            merges.insert((ida, idb), id);
            str_to_id.insert(format!("{a}{b}"), id);
            let mut bytes = vocab[&ida].clone();
            bytes.extend_from_slice(&vocab[&idb]);
            vocab.insert(id, bytes);
            rank += 1;
        }
        Ok(Gpt2Tokenizer {
            merges,
            vocab,
            byte_to_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + 1
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk in split(text, SplitPattern::Gpt2) {
            let shuffled: Vec<u32> = chunk
                .bytes()
                .map(|b| self.byte_to_id[b as usize])
                .collect();
            ids.extend(encode_ids(shuffled, &self.merges));
        }
        ids
    }

    pub fn encode(&self, text: &str, allow_special: bool) -> Vec<u32> {
        if !allow_special || !text.contains(ENDOFTEXT) {
            return self.encode_ordinary(text);
        }
        let mut out = Vec::new();
        let mut remainder = text;
        while let Some(pos) = remainder.find(ENDOFTEXT) {
            if pos > 0 {
                out.extend(self.encode_ordinary(&remainder[..pos]));
            }
            out.push(ENDOFTEXT_ID);
            remainder = &remainder[pos + ENDOFTEXT.len()..];
        }
        if !remainder.is_empty() {
            out.extend(self.encode_ordinary(remainder));
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id == ENDOFTEXT_ID {
                bytes.extend_from_slice(ENDOFTEXT.as_bytes());
            } else if let Some(t) = self.vocab.get(&id) {
                bytes.extend_from_slice(t);
            } else {
                panic!("invalid token id: {id}");
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.vocab.get(&id).map(|v| v.as_slice())
    }
}
