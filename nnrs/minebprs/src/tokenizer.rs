use std::collections::HashMap;
use std::io::{self, Read, Write};

pub type Pair = (u32, u32);

#[derive(Default)]
pub struct Stats {
    index: HashMap<Pair, usize>,
    entries: Vec<(Pair, u64)>,
}

impl Stats {
    pub fn new() -> Stats {
        Stats::default()
    }

    pub fn update(&mut self, ids: &[u32]) {
        for w in ids.windows(2) {
            let p = (w[0], w[1]);
            match self.index.get(&p) {
                Some(&i) => self.entries[i].1 += 1,
                None => {
                    self.index.insert(p, self.entries.len());
                    self.entries.push((p, 1));
                }
            }
        }
    }

    pub fn argmax(&self) -> Option<(Pair, u64)> {
        let mut best: Option<(Pair, u64)> = None;
        for &(p, c) in &self.entries {
            if best.map_or(true, |(_, bc)| c > bc) {
                best = Some((p, c));
            }
        }
        best
    }
}

pub fn merge(ids: &[u32], pair: Pair, idx: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(idx);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

#[derive(Default, Clone)]
pub struct Merges {
    pub list: Vec<Pair>,
    pub map: HashMap<Pair, u32>,
}

impl Merges {
    pub fn push(&mut self, pair: Pair) -> u32 {
        let id = 256 + self.list.len() as u32;
        self.list.push(pair);
        self.map.insert(pair, id);
        id
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}

pub fn encode_ids(mut ids: Vec<u32>, merges: &HashMap<Pair, u32>) -> Vec<u32> {
    while ids.len() >= 2 {
        let mut best: Option<(u32, Pair)> = None;
        for w in ids.windows(2) {
            let p = (w[0], w[1]);
            if let Some(&id) = merges.get(&p) {
                if best.map_or(true, |(bid, _)| id < bid) {
                    best = Some((id, p));
                }
            }
        }
        match best {
            Some((idx, pair)) => ids = merge(&ids, pair, idx),
            None => break,
        }
    }
    ids
}

pub fn encode_chunk(bytes: &[u8], merges: &HashMap<Pair, u32>) -> Vec<u32> {
    encode_ids(bytes.iter().map(|&b| b as u32).collect(), merges)
}

pub fn build_vocab(merges: &Merges) -> HashMap<u32, Vec<u8>> {
    let mut vocab: HashMap<u32, Vec<u8>> = (0..256u32).map(|i| (i, vec![i as u8])).collect();
    for (r, &(a, b)) in merges.list.iter().enumerate() {
        let mut t = vocab[&a].clone();
        t.extend_from_slice(&vocab[&b]);
        vocab.insert(256 + r as u32, t);
    }
    vocab
}

pub fn render_token(t: &[u8]) -> String {
    String::from_utf8_lossy(t)
        .chars()
        .map(|c| {
            if c.is_control() {
                format!("\\u{:04x}", c as u32)
            } else {
                c.to_string()
            }
        })
        .collect()
}

pub fn save_model(
    file_prefix: &str,
    pattern: &str,
    special_tokens: &HashMap<String, u32>,
    merges: &Merges,
) -> io::Result<()> {
    let mut f = std::fs::File::create(format!("{file_prefix}.model"))?;
    writeln!(f, "minbpe v1")?;
    writeln!(f, "{pattern}")?;
    writeln!(f, "{}", special_tokens.len())?;
    let mut specials: Vec<(&String, &u32)> = special_tokens.iter().collect();
    specials.sort_by_key(|(_, &id)| id);
    for (tok, id) in specials {
        writeln!(f, "{tok} {id}")?;
    }
    for &(a, b) in &merges.list {
        writeln!(f, "{a} {b}")?;
    }

    let vocab = build_vocab(merges);
    let mut vf = std::fs::File::create(format!("{file_prefix}.vocab"))?;
    let inverted: HashMap<u32, Pair> = merges.map.iter().map(|(&p, &id)| (id, p)).collect();
    let mut ids: Vec<&u32> = vocab.keys().collect();
    ids.sort();
    for &id in ids {
        let s = render_token(&vocab[&id]);
        match inverted.get(&id) {
            Some(&(a, b)) => writeln!(
                vf,
                "[{}][{}] -> [{}] {}",
                render_token(&vocab[&a]),
                render_token(&vocab[&b]),
                s,
                id
            )?,
            None => writeln!(vf, "[{s}] {id}")?,
        }
    }
    Ok(())
}

pub fn load_model(path: &str) -> io::Result<(String, HashMap<String, u32>, Merges)> {
    let mut text = String::new();
    std::fs::File::open(path)?.read_to_string(&mut text)?;
    let mut lines = text.lines();
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    if lines.next().ok_or_else(|| bad("empty file"))? != "minbpe v1" {
        return Err(bad("not a minbpe v1 model file"));
    }
    let pattern = lines
        .next()
        .ok_or_else(|| bad("missing pattern line"))?
        .to_string();
    let num_special: usize = lines
        .next()
        .ok_or_else(|| bad("missing special count"))?
        .trim()
        .parse()
        .map_err(|_| bad("bad special count"))?;
    let mut special_tokens = HashMap::new();
    for _ in 0..num_special {
        let line = lines
            .next()
            .ok_or_else(|| bad("missing special token line"))?;
        let (tok, id) = line
            .rsplit_once(' ')
            .ok_or_else(|| bad("bad special token line"))?;
        special_tokens.insert(
            tok.to_string(),
            id.trim().parse().map_err(|_| bad("bad special id"))?,
        );
    }
    let mut merges = Merges::default();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let a: u32 = it
            .next()
            .ok_or_else(|| bad("bad merge line"))?
            .parse()
            .map_err(|_| bad("bad merge id"))?;
        let b: u32 = it
            .next()
            .ok_or_else(|| bad("bad merge line"))?
            .parse()
            .map_err(|_| bad("bad merge id"))?;
        merges.push((a, b));
    }
    Ok((pattern, special_tokens, merges))
}
