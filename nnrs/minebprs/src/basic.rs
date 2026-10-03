use std::collections::HashMap;
use std::io;

use crate::tokenizer::{build_vocab, encode_chunk, merge, render_token, Merges, Stats};

pub struct BasicTokenizer {
    pub merges: Merges,
    pub vocab: HashMap<u32, Vec<u8>>,
}

impl Default for BasicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicTokenizer {
    pub fn new() -> BasicTokenizer {
        BasicTokenizer {
            merges: Merges::default(),
            vocab: build_vocab(&Merges::default()),
        }
    }

    pub fn train(&mut self, text: &str, vocab_size: usize, verbose: bool) {
        assert!(vocab_size >= 256);
        let num_merges = vocab_size - 256;
        let mut ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        let mut merges = Merges::default();
        for i in 0..num_merges {
            let mut stats = Stats::new();
            stats.update(&ids);
            let Some((pair, count)) = stats.argmax() else {
                break;
            };
            let idx = merges.push(pair);
            ids = merge(&ids, pair, idx);
            if verbose {
                let vocab = build_vocab(&merges);
                println!(
                    "merge {}/{}: {:?} -> {} ({}) had {} occurrences",
                    i + 1,
                    num_merges,
                    pair,
                    idx,
                    render_token(&vocab[&idx]),
                    count
                );
            }
        }
        self.vocab = build_vocab(&merges);
        self.merges = merges;
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        encode_chunk(text.as_bytes(), &self.merges.map)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(
                self.vocab
                    .get(&id)
                    .unwrap_or_else(|| panic!("invalid token id: {id}")),
            );
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn save(&self, file_prefix: &str) -> io::Result<()> {
        crate::tokenizer::save_model(file_prefix, "", &HashMap::new(), &self.merges)
    }

    pub fn load(path: &str) -> io::Result<BasicTokenizer> {
        let (pattern, _specials, merges) = crate::tokenizer::load_model(path)?;
        if !pattern.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "model has a split pattern; load it as a RegexTokenizer",
            ));
        }
        Ok(BasicTokenizer {
            vocab: build_vocab(&merges),
            merges,
        })
    }
}
