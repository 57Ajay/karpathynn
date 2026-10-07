use minebprs::basic::BasicTokenizer;
use minebprs::gpt2::Gpt2Tokenizer;
use minebprs::rgx::{AllowedSpecial, RegexTokenizer};
use minebprs::split::SplitPattern;
use minebprs::tokenizer::render_token;

const USAGE: &str = "minebprs: byte-level BPE tokenizer

USAGE:
  minebprs train  --input <file> --vocab-size <n> [--type basic|regex]
                  [--pattern gpt2|gpt4] [--output <prefix>] [--verbose]
  minebprs encode --model <x.model> (--text <s> | --input <file>)
  minebprs decode --model <x.model> --ids \"31373 995\"
  minebprs show   --model <x.model> --text <s>
  minebprs gpt2   (--text <s> | --input <file> | --ids \"...\")
                  [--vocab-bpe gpt2/vocab.bpe] [--allow-special] [--show]
";

fn get_flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| {
        args.get(i + 1)
            .unwrap_or_else(|| panic!("missing value for {name}"))
            .clone()
    })
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn read_text(args: &[String]) -> String {
    if let Some(t) = get_flag(args, "--text") {
        t
    } else if let Some(f) = get_flag(args, "--input") {
        std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("cannot read {f}: {e}"))
    } else {
        panic!("provide --text or --input\n{USAGE}");
    }
}

fn parse_ids(s: &str) -> Vec<u32> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().unwrap_or_else(|_| panic!("bad token id: {t}")))
        .collect()
}

enum Loaded {
    Basic(BasicTokenizer),
    Rgx(RegexTokenizer),
}

impl Loaded {
    fn open(path: &str) -> Loaded {
        let (pattern, _, _) = minebprs::tokenizer::load_model(path).expect("cannot read model");
        if pattern.is_empty() {
            Loaded::Basic(BasicTokenizer::load(path).unwrap())
        } else {
            Loaded::Rgx(RegexTokenizer::load(path).unwrap())
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Loaded::Basic(t) => t.encode(text),
            Loaded::Rgx(t) => t.encode(text, AllowedSpecial::All),
        }
    }

    fn decode(&self, ids: &[u32]) -> String {
        match self {
            Loaded::Basic(t) => t.decode(ids),
            Loaded::Rgx(t) => t.decode(ids),
        }
    }

    fn token_bytes(&self, id: u32) -> Vec<u8> {
        let vocab = match self {
            Loaded::Basic(t) => &t.vocab,
            Loaded::Rgx(t) => &t.vocab,
        };
        vocab.get(&id).cloned().unwrap_or_default()
    }
}

fn show_tokens(ids: &[u32], bytes_of: impl Fn(u32) -> Vec<u8>) {
    let rendered: Vec<String> = ids
        .iter()
        .map(|&id| format!("[{}]", render_token(&bytes_of(id))))
        .collect();
    println!("{}", rendered.join(""));
    let pairs: Vec<String> = ids.iter().map(|&id| id.to_string()).collect();
    println!("{} tokens: {}", ids.len(), pairs.join(" "));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    match cmd {
        "train" => {
            let input = get_flag(&args, "--input").expect("--input required");
            let vocab_size: usize = get_flag(&args, "--vocab-size")
                .expect("--vocab-size required")
                .parse()
                .unwrap();
            let kind = get_flag(&args, "--type").unwrap_or_else(|| "regex".into());
            let output = get_flag(&args, "--output").unwrap_or_else(|| "models/tok".into());
            let verbose = has_flag(&args, "--verbose");
            let text = std::fs::read_to_string(&input).expect("cannot read input");
            if let Some(dir) = std::path::Path::new(&output).parent() {
                std::fs::create_dir_all(dir).ok();
            }
            let t0 = std::time::Instant::now();
            match kind.as_str() {
                "basic" => {
                    let mut t = BasicTokenizer::new();
                    t.train(&text, vocab_size, verbose);
                    t.save(&output).unwrap();
                }
                "regex" => {
                    let pattern = match get_flag(&args, "--pattern").as_deref() {
                        Some("gpt2") => SplitPattern::Gpt2,
                        _ => SplitPattern::Gpt4,
                    };
                    let mut t = RegexTokenizer::new(pattern);
                    t.train(&text, vocab_size, verbose);
                    t.save(&output).unwrap();
                }
                other => panic!("unknown --type {other} (basic|regex)"),
            }
            println!(
                "trained vocab_size={vocab_size} on {} bytes in {:.2}s -> {output}.model",
                text.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        "encode" => {
            let tok = Loaded::open(&get_flag(&args, "--model").expect("--model required"));
            let ids = tok.encode(&read_text(&args));
            println!(
                "{}",
                ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            eprintln!("({} tokens)", ids.len());
        }
        "decode" => {
            let tok = Loaded::open(&get_flag(&args, "--model").expect("--model required"));
            println!(
                "{}",
                tok.decode(&parse_ids(
                    &get_flag(&args, "--ids").expect("--ids required")
                ))
            );
        }
        "show" => {
            let tok = Loaded::open(&get_flag(&args, "--model").expect("--model required"));
            let ids = tok.encode(&read_text(&args));
            show_tokens(&ids, |id| tok.token_bytes(id));
        }
        "gpt2" => {
            let path = get_flag(&args, "--vocab-bpe").unwrap_or_else(|| "gpt2/vocab.bpe".into());
            let tok = Gpt2Tokenizer::load(&path).expect("cannot load vocab.bpe");
            if let Some(ids) = get_flag(&args, "--ids") {
                println!("{}", tok.decode(&parse_ids(&ids)));
                return;
            }
            let text = read_text(&args);
            let ids = tok.encode(&text, has_flag(&args, "--allow-special"));
            if has_flag(&args, "--show") {
                show_tokens(&ids, |id| {
                    tok.token_bytes(id)
                        .map(|b| b.to_vec())
                        .unwrap_or_else(|| b"<|endoftext|>".to_vec())
                });
            } else {
                println!(
                    "{}",
                    ids.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                eprintln!("({} tokens, vocab {})", ids.len(), tok.vocab_size());
            }
        }
        "--help" | "-h" | "help" => print!("{USAGE}"),
        other => {
            eprintln!("unknown command: {other:?}\n{USAGE}");
            std::process::exit(1);
        }
    }
}
