use std::collections::HashMap;

use minebprs::basic::BasicTokenizer;
use minebprs::rgx::{AllowedSpecial, RegexTokenizer};
use minebprs::split::SplitPattern;

#[test]
fn test_wikipedia_example() {
    let mut t = BasicTokenizer::new();
    t.train("aaabdaaabac", 256 + 3, false);
    assert_eq!(t.merges.list, vec![(97, 97), (256, 97), (257, 98)]);
    assert_eq!(t.encode("aaabdaaabac"), vec![258, 100, 258, 97, 99]);
    assert_eq!(t.decode(&[258, 100, 258, 97, 99]), "aaabdaaabac");
}

const TRICKY: [&str; 7] = [
    "hello world!!!? (안녕하세요!) lol123 😉",
    "",
    "?",
    "a",
    "FILE:taylorswift.txt",
    "tab\tnewline\nmixed  spaces   ",
    "🚀🚀🚀 unicode ½ ⅓ Ⅻ? ok",
];

fn cases() -> Vec<String> {
    let ts = std::fs::read_to_string("tests/taylorswift.txt").unwrap();
    TRICKY
        .iter()
        .map(|s| {
            if *s == "FILE:taylorswift.txt" {
                ts.clone()
            } else {
                s.to_string()
            }
        })
        .collect()
}

#[test]
fn test_roundtrips() {
    let ts = std::fs::read_to_string("tests/taylorswift.txt").unwrap();
    let train_slice = &ts[..20_000.min(ts.len())];

    let mut basic = BasicTokenizer::new();
    basic.train(train_slice, 300, false);
    let mut rgx2 = RegexTokenizer::new(SplitPattern::Gpt2);
    rgx2.train(train_slice, 300, false);
    let mut rgx4 = RegexTokenizer::new(SplitPattern::Gpt4);
    rgx4.train(train_slice, 300, false);
    let untrained = BasicTokenizer::new();

    for text in cases() {
        assert_eq!(basic.decode(&basic.encode(&text)), text);
        assert_eq!(rgx2.decode(&rgx2.encode_ordinary(&text)), text);
        assert_eq!(rgx4.decode(&rgx4.encode_ordinary(&text)), text);
        assert_eq!(untrained.decode(&untrained.encode(&text)), text);
    }
}

#[test]
fn test_trained_tokenizer_compresses() {
    let ts = std::fs::read_to_string("tests/taylorswift.txt").unwrap();
    let mut t = RegexTokenizer::new(SplitPattern::Gpt4);
    t.train(&ts, 512, false);
    let n_tokens = t.encode_ordinary(&ts).len();
    let n_bytes = ts.len();
    assert!(
        (n_tokens as f64) < 0.75 * n_bytes as f64,
        "expected >1.33x compression, got {n_bytes} bytes -> {n_tokens} tokens"
    );
}

#[test]
fn test_save_load_roundtrip() {
    let dir = std::env::temp_dir().join("minebpe_test");
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("tok").to_string_lossy().into_owned();

    let mut t = RegexTokenizer::new(SplitPattern::Gpt4);
    t.train(
        &std::fs::read_to_string("tests/taylorswift.txt").unwrap()[..20_000],
        300,
        false,
    );
    t.register_special_tokens(HashMap::from([("<|endoftext|>".to_string(), 300u32)]));
    t.save(&prefix).unwrap();

    let loaded = RegexTokenizer::load(&format!("{prefix}.model")).unwrap();
    assert_eq!(loaded.merges.list, t.merges.list);
    assert_eq!(loaded.special_tokens, t.special_tokens);
    for text in cases() {
        assert_eq!(loaded.encode_ordinary(&text), t.encode_ordinary(&text));
    }

    let mut b = BasicTokenizer::new();
    b.train("aaabdaaabac", 259, false);
    let bprefix = dir.join("btok").to_string_lossy().into_owned();
    b.save(&bprefix).unwrap();
    let bl = BasicTokenizer::load(&format!("{bprefix}.model")).unwrap();
    assert_eq!(bl.encode("aaabdaaabac"), b.encode("aaabdaaabac"));
}

#[test]
fn test_special_tokens_handling() {
    let mut t = RegexTokenizer::new(SplitPattern::Gpt4);
    t.register_special_tokens(HashMap::from([
        ("<|endoftext|>".to_string(), 1000),
        ("<|im_start|>".to_string(), 1001),
        ("<|im_end|>".to_string(), 1002),
    ]));

    let text = "<|im_start|>user\nhello<|im_end|>";
    assert_eq!(
        t.encode(text, AllowedSpecial::All),
        vec![1001, 117, 115, 101, 114, 10, 104, 101, 108, 108, 111, 1002]
    );
    assert_eq!(
        t.decode(&t.encode(text, AllowedSpecial::All)),
        text
    );
}
