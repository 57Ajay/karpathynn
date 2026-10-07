use minebprs::gpt2::{Gpt2Tokenizer, ENDOFTEXT_ID};

include!("gpt2_fixtures.rs");

fn load() -> Gpt2Tokenizer {
    Gpt2Tokenizer::load("gpt2/vocab.bpe").expect("gpt2/vocab.bpe missing")
}

#[test]
fn test_byte_identical_to_reference_gpt2() {
    let tok = load();
    for (text, expected) in CASES {
        let got = tok.encode_ordinary(text);
        assert_eq!(&got, expected, "GPT-2 id mismatch on {text:?}");
    }
}

#[test]
fn test_decode_inverts_encode() {
    let tok = load();
    for (text, _) in CASES {
        let ids = tok.encode_ordinary(text);
        assert_eq!(tok.decode(&ids), *text);
    }
}

#[test]
fn test_karpathy_sanity_string() {
    let tok = load();
    assert_eq!(
        tok.encode_ordinary("Hello, I'm a language model,"),
        vec![15496, 11, 314, 1101, 257, 3303, 2746, 11]
    );
}

#[test]
fn test_endoftext_special_token() {
    let tok = load();
    let ids = tok.encode("hello <|endoftext|>", true);
    let mut expected = tok.encode_ordinary("hello ");
    expected.push(ENDOFTEXT_ID);
    assert_eq!(ids, expected);
    assert_eq!(tok.decode(&ids), "hello <|endoftext|>");

    let plain = tok.encode("hello <|endoftext|>", false);
    assert!(!plain.contains(&ENDOFTEXT_ID));
    assert_eq!(tok.decode(&plain), "hello <|endoftext|>");
}

#[test]
fn test_vocab_is_complete_gpt2() {
    let tok = load();
    assert_eq!(tok.vocab_size(), 50257);
}
