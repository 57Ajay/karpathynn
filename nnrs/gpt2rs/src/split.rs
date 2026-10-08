pub const GPT2_SPLIT_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
pub const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SplitPattern {
    Gpt2,
    Gpt4,
}

impl SplitPattern {
    pub fn as_str(&self) -> &'static str {
        match self {
            SplitPattern::Gpt2 => GPT2_SPLIT_PATTERN,
            SplitPattern::Gpt4 => GPT4_SPLIT_PATTERN,
        }
    }

    pub fn from_pattern_str(s: &str) -> Option<SplitPattern> {
        match s {
            GPT2_SPLIT_PATTERN => Some(SplitPattern::Gpt2),
            GPT4_SPLIT_PATTERN => Some(SplitPattern::Gpt4),
            _ => None,
        }
    }
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !c.is_numeric()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn is_ws(c: char) -> bool {
    c.is_whitespace()
}

fn is_other(c: char) -> bool {
    !is_ws(c) && !is_letter(c) && !is_number(c)
}

pub fn split(text: &str, pat: SplitPattern) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let j = match pat {
            SplitPattern::Gpt2 => match_gpt2(&chars, i),
            SplitPattern::Gpt4 => match_gpt4(&chars, i),
        };
        debug_assert!(j > i, "scanner must always advance");
        let start = chars[i].0;
        let end = if j < n { chars[j].0 } else { text.len() };
        out.push(&text[start..end]);
        i = j;
    }
    out
}

fn contraction_len(chars: &[(usize, char)], i: usize, case_insensitive: bool) -> Option<usize> {
    let n = chars.len();
    if chars[i].1 != '\'' || i + 1 >= n {
        return None;
    }
    let fold = |c: char| {
        if case_insensitive {
            c.to_ascii_lowercase()
        } else {
            c
        }
    };
    let c1 = fold(chars[i + 1].1);
    if i + 2 < n {
        let c2 = fold(chars[i + 2].1);
        if matches!((c1, c2), ('l', 'l') | ('v', 'e') | ('r', 'e')) {
            return Some(3);
        }
    }
    if matches!(c1, 's' | 'd' | 'm' | 't') {
        return Some(2);
    }
    None
}

fn match_gpt2(chars: &[(usize, char)], i: usize) -> usize {
    let n = chars.len();
    let c = chars[i].1;

    if let Some(len) = contraction_len(chars, i, false) {
        return i + len;
    }

    {
        let k = if c == ' ' { i + 1 } else { i };
        if k < n && is_letter(chars[k].1) {
            let mut j = k + 1;
            while j < n && is_letter(chars[j].1) {
                j += 1;
            }
            return j;
        }
    }

    {
        let k = if c == ' ' { i + 1 } else { i };
        if k < n && is_number(chars[k].1) {
            let mut j = k + 1;
            while j < n && is_number(chars[j].1) {
                j += 1;
            }
            return j;
        }
    }

    {
        let k = if c == ' ' { i + 1 } else { i };
        if k < n && is_other(chars[k].1) {
            let mut j = k + 1;
            while j < n && is_other(chars[j].1) {
                j += 1;
            }
            return j;
        }
    }

    debug_assert!(is_ws(c), "rules 1-4 cover every non-whitespace char");
    let mut r = i + 1;
    while r < n && is_ws(chars[r].1) {
        r += 1;
    }
    if r == n {
        return r;
    }
    if r - 1 > i {
        return r - 1;
    }
    i + 1
}

fn match_gpt4(chars: &[(usize, char)], i: usize) -> usize {
    let n = chars.len();
    let c = chars[i].1;

    if let Some(len) = contraction_len(chars, i, true) {
        return i + len;
    }

    {
        let takes_prefix = c != '\r' && c != '\n' && !is_letter(c) && !is_number(c);
        let k = if takes_prefix { i + 1 } else { i };
        if k < n && is_letter(chars[k].1) {
            let mut j = k + 1;
            while j < n && is_letter(chars[j].1) {
                j += 1;
            }
            return j;
        }
    }

    if is_number(c) {
        let mut j = i + 1;
        while j < n && j < i + 3 && is_number(chars[j].1) {
            j += 1;
        }
        return j;
    }

    {
        let k = if c == ' ' { i + 1 } else { i };
        if k < n && is_other(chars[k].1) {
            let mut j = k + 1;
            while j < n && is_other(chars[j].1) {
                j += 1;
            }
            while j < n && matches!(chars[j].1, '\r' | '\n') {
                j += 1;
            }
            return j;
        }
    }

    debug_assert!(is_ws(c));
    let mut r = i + 1;
    while r < n && is_ws(chars[r].1) {
        r += 1;
    }

    {
        let mut last_nl = None;
        for (off, &(_, ch)) in chars[i..r].iter().enumerate() {
            if matches!(ch, '\r' | '\n') {
                last_nl = Some(i + off);
            }
        }
        if let Some(m) = last_nl {
            return m + 1;
        }
    }

    if r == n {
        return r;
    }
    if r - 1 > i {
        return r - 1;
    }

    i + 1
}
