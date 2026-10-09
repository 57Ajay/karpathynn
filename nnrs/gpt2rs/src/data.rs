use crate::bpe_gpt2::Gpt2Tokenizer;
use crate::rng::Rng;

pub struct CharTok {
    pub chars: Vec<char>,
}

impl CharTok {
    pub fn from_text(text: &str) -> CharTok {
        let mut chars: Vec<char> = {
            let set: std::collections::HashSet<char> = text.chars().collect();
            set.into_iter().collect()
        };
        chars.sort();
        CharTok { chars }
    }

    pub fn from_chars(chars: Vec<char>) -> CharTok {
        CharTok { chars }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    pub fn encode(&self, s: &str) -> Vec<u32> {
        s.chars()
            .map(|c| {
                self.chars
                    .binary_search(&c)
                    .unwrap_or_else(|_| panic!("char {c:?} not in vocabulary"))
                    as u32
            })
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.chars[i as usize]).collect()
    }
}

pub enum Tok {
    Char(CharTok),
    Bpe(Box<Gpt2Tokenizer>),
}

impl Tok {
    pub fn encode(&self, s: &str) -> Vec<u32> {
        match self {
            Tok::Char(t) => t.encode(s),
            Tok::Bpe(t) => t.encode_ordinary(s),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Tok::Char(t) => t.decode(ids),
            Tok::Bpe(t) => t.decode(ids),
        }
    }
}

pub struct Dataset {
    pub ids: Vec<u32>,
}

impl Dataset {
    pub fn new(ids: Vec<u32>) -> Dataset {
        Dataset { ids }
    }

    pub fn split(&self) -> (&[u32], &[u32]) {
        let n = self.ids.len() * 9 / 10;
        (&self.ids[..n], &self.ids[n..])
    }

    pub fn sample_batch(
        split: &[u32],
        rng: &mut Rng,
        batch: usize,
        block: usize,
    ) -> (Vec<u32>, Vec<i32>) {
        let mut x = Vec::with_capacity(batch * block);
        let mut y = Vec::with_capacity(batch * block);
        for _ in 0..batch {
            let o = rng.below(split.len() - block - 1);
            x.extend_from_slice(&split[o..o + block]);
            y.extend(split[o + 1..o + block + 1].iter().map(|&t| t as i32));
        }
        (x, y)
    }
}
