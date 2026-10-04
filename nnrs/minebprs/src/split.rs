pub const GPT2_SPLIT_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
pub const GPT4_SPLIT_PATTERN: &str =
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

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
        debug_assert!(j > i);
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

fn take_while(chars: &[(usize, char)], mut i: usize, pred: impl Fn(char) -> bool) -> usize {
    while i < chars.len() && pred(chars[i].1) {
        i += 1;
    }
    i
}

fn match_gpt2(chars: &[(usize, char)], i: usize) -> usize {
    let n = chars.len();

    if let Some(len) = contraction_len(chars, i, false) {
        return i + len;
    }

    let (has_space, after_sp) = if chars[i].1 == ' ' && i + 1 < n {
        (true, i + 1)
    } else {
        (false, i)
    };

    if is_letter(chars[after_sp].1) {
        return take_while(chars, after_sp + 1, is_letter);
    }
    if is_number(chars[after_sp].1) {
        return take_while(chars, after_sp + 1, is_number);
    }
    if is_other(chars[after_sp].1) {
        return take_while(chars, after_sp + 1, is_other);
    }

    if chars[i].1 == ' ' && !has_space {
        let j = take_while(chars, i + 1, |c| c == ' ');
        if j < n && !is_ws(chars[j].1) {
            return (j - 1).max(i + 1);
        }
        return j;
    }

    if is_ws(chars[i].1) {
        let j = take_while(chars, i + 1, is_ws);
        if j < n && !is_ws(chars[j].1) {
            return (j - 1).max(i + 1);
        }
        return j;
    }

    take_while(chars, i + 1, is_ws)
}

fn is_gpt4_prefix(c: char) -> bool {
    c != '\r' && c != '\n' && !is_letter(c) && !is_number(c)
}

fn match_gpt4(chars: &[(usize, char)], i: usize) -> usize {
    let n = chars.len();

    if let Some(len) = contraction_len(chars, i, true) {
        return i + len;
    }

    let letter_start = if is_gpt4_prefix(chars[i].1) && i + 1 < n {
        i + 1
    } else {
        i
    };
    if is_letter(chars[letter_start].1) {
        return take_while(chars, letter_start + 1, is_letter);
    }

    if is_number(chars[i].1) {
        let mut j = i + 1;
        while j < n && j < i + 3 && is_number(chars[j].1) {
            j += 1;
        }
        return j;
    }

    let punct_start = if chars[i].1 == ' ' && i + 1 < n {
        i + 1
    } else {
        i
    };
    if is_other(chars[punct_start].1) {
        let j = take_while(chars, punct_start + 1, is_other);
        return take_while(chars, j, |c| c == '\r' || c == '\n');
    }

    debug_assert!(is_ws(chars[i].1));
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
