use std::collections::HashMap;
use std::io;

use crate::split::{split, SplitPattern};
use crate::tokenizer::{build_vocab, encode_chunk, merge, render_token, Merges, Stats};

pub enum AllowedSpecial<'a> {
    All,
    None,
    NoneRaise,
    Set(&'a [&'a str]),
}

pub struct RegexTokenizer {
    pub pattern: SplitPattern,
    pub merges: Merges,
    pub vocab: HashMap<u32, Vec<u8>>,
    pub special_tokens: HashMap<String, u32>,
    inverse_special: HashMap<u32, String>,
}

impl RegexTokenizer {
    pub fn new(pattern: SplitPattern) -> RegexTokenizer {
        RegexTokenizer {
            pattern,
            merges: Merges::default(),
            vocab: build_vocab(&Merges::default()),
            special_tokens: HashMap::new(),
            inverse_special: HashMap::new(),
        }
    }

    pub fn train(&mut self, text: &str, vocab_size: usize, verbose: bool) {
        assert!(vocab_size >= 256);
        let num_merges = vocab_size - 256;
        let mut ids: Vec<Vec<u32>> = split(text, self.pattern)
            .iter()
            .map(|ch| ch.bytes().map(|b| b as u32).collect())
            .collect();
        let mut merges = Merges::default();
        for i in 0..num_merges {
            let mut stats = Stats::new();
            for chunk in &ids {
                stats.update(chunk);
            }
            let Some((pair, count)) = stats.argmax() else {
                break;
            };
            let idx = merges.push(pair);
            for chunk in ids.iter_mut() {
                *chunk = merge(chunk, pair, idx);
            }
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

    pub fn register_special_tokens(&mut self, special_tokens: HashMap<String, u32>) {
        self.inverse_special = special_tokens
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();
        self.special_tokens = special_tokens;
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk in split(text, self.pattern) {
            ids.extend(encode_chunk(chunk.as_bytes(), &self.merges.map));
        }
        ids
    }

    pub fn encode(&self, text: &str, allowed: AllowedSpecial) -> Vec<u32> {
        let special: HashMap<&str, u32> = match allowed {
            AllowedSpecial::All => self
                .special_tokens
                .iter()
                .map(|(k, &v)| (k.as_str(), v))
                .collect(),
            AllowedSpecial::None => HashMap::new(),
            AllowedSpecial::NoneRaise => {
                for tok in self.special_tokens.keys() {
                    assert!(
                        !text.contains(tok),
                        "text contains special token {tok:?} but allowed=NoneRaise"
                    );
                }
                HashMap::new()
            }
            AllowedSpecial::Set(allowed_list) => {
                let mut m = HashMap::new();
                for &tok in allowed_list {
                    if let Some(&id) = self.special_tokens.get(tok) {
                        m.insert(tok, id);
                    }
                }
                m
            }
        };

        if special.is_empty() {
            return self.encode_ordinary(text);
        }

        let mut out = Vec::new();
        let mut remainder = text;
        while !remainder.is_empty() {
            let mut earliest: Option<(usize, &str, u32)> = None;
            for (&tok, &id) in &special {
                if let Some(pos) = remainder.find(tok) {
                    if earliest.map_or(true, |(epos, _, _)| pos < epos) {
                        earliest = Some((pos, tok, id));
                    }
                }
            }
            match earliest {
                Some((pos, tok, id)) => {
                    if pos > 0 {
                        out.extend(self.encode_ordinary(&remainder[..pos]));
                    }
                    out.push(id);
                    remainder = &remainder[pos + tok.len()..];
                }
                None => {
                    out.extend(self.encode_ordinary(remainder));
                    break;
                }
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(t) = self.vocab.get(&id) {
                bytes.extend_from_slice(t);
            } else if let Some(s) = self.inverse_special.get(&id) {
                bytes.extend_from_slice(s.as_bytes());
            } else {
                panic!("invalid token id: {id}");
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn save(&self, file_prefix: &str) -> io::Result<()> {
        crate::tokenizer::save_model(
            file_prefix,
            self.pattern.as_str(),
            &self.special_tokens,
            &self.merges,
        )
    }

    pub fn load(path: &str) -> io::Result<RegexTokenizer> {
        let (pat_str, special_tokens, merges) = crate::tokenizer::load_model(path)?;
        let pattern = SplitPattern::from_pattern_str(&pat_str).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unrecognized split pattern: {pat_str}"),
            )
        })?;
        let inverse_special = special_tokens
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();
        let vocab = build_vocab(&merges);
        Ok(RegexTokenizer {
            pattern,
            merges,
            vocab,
            special_tokens,
            inverse_special,
        })
    }
}
