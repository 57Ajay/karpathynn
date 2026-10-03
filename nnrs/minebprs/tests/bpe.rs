use minebprs::basic::BasicTokenizer;

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
    let untrained = BasicTokenizer::new();

    for text in cases() {
        assert_eq!(basic.decode(&basic.encode(&text)), text);
        assert_eq!(untrained.decode(&untrained.encode(&text)), text);
    }
}

#[test]
fn test_save_load_roundtrip() {
    let dir = std::env::temp_dir().join("minebpe_test");
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("btok").to_string_lossy().into_owned();

    let mut b = BasicTokenizer::new();
    b.train("aaabdaaabac", 259, false);
    b.save(&prefix).unwrap();
    let bl = BasicTokenizer::load(&format!("{prefix}.model")).unwrap();
    assert_eq!(bl.encode("aaabdaaabac"), b.encode("aaabdaaabac"));
}
