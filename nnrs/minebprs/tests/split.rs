use minebprs::split::{split, SplitPattern};

include!("split_fixtures.rs");

#[test]
fn test_gpt2_pattern_matches_regex_engine() {
    for (text, expected) in SPLIT_GPT2 {
        let got = split(text, SplitPattern::Gpt2);
        assert_eq!(&got, expected);
    }
}

#[test]
fn test_gpt4_pattern_matches_regex_engine() {
    for (text, expected) in SPLIT_GPT4 {
        let got = split(text, SplitPattern::Gpt4);
        assert_eq!(&got, expected);
    }
}

#[test]
fn test_split_covers_text_exactly() {
    let ts = std::fs::read_to_string("tests/taylorswift.txt").unwrap();
    for pat in [SplitPattern::Gpt2, SplitPattern::Gpt4] {
        let joined: String = split(&ts, pat).concat();
        assert_eq!(joined, ts);
    }
}
