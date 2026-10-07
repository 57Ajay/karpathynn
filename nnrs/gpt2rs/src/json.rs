use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(HashMap<String, Json>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) => Some(*n as usize),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&HashMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }
}

pub fn parse(src: &str) -> Result<Json, String> {
    let bytes: Vec<char> = src.chars().collect();
    let mut p = Parser { c: &bytes, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.c.len() {
        return Err(format!("trailing garbage at char {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    c: &'a [char],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.c.len() && self.c[self.i].is_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.c.get(self.i).copied()
    }

    fn eat(&mut self, ch: char) -> Result<(), String> {
        if self.peek() == Some(ch) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {ch:?} at char {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some('t') => self.lit("true", Json::Bool(true)),
            Some('f') => self.lit("false", Json::Bool(false)),
            Some('n') => self.lit("null", Json::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            other => Err(format!("unexpected {other:?} at char {}", self.i)),
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        for ch in word.chars() {
            self.eat(ch)?;
        }
        Ok(v)
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat('{')?;
        let mut m = HashMap::new();
        self.ws();
        if self.peek() == Some('}') {
            self.i += 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.eat(':')?;
            self.ws();
            m.insert(k, self.value()?);
            self.ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some('}') => {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                other => return Err(format!("expected , or }} got {other:?}")),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat('[')?;
        let mut v = Vec::new();
        self.ws();
        if self.peek() == Some(']') {
            self.i += 1;
            return Ok(Json::Arr(v));
        }
        loop {
            self.ws();
            v.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some(']') => {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                other => return Err(format!("expected , or ] got {other:?}")),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat('"')?;
        let mut s = String::new();
        loop {
            let ch = self.peek().ok_or("unterminated string")?;
            self.i += 1;
            match ch {
                '"' => return Ok(s),
                '\\' => {
                    let esc = self.peek().ok_or("bad escape")?;
                    self.i += 1;
                    match esc {
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        '/' => s.push('/'),
                        'b' => s.push('\u{8}'),
                        'f' => s.push('\u{c}'),
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        'u' => {
                            let hi = self.hex4()?;

                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                self.eat('\\')?;
                                self.eat('u')?;
                                let lo = self.hex4()?;
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            s.push(char::from_u32(cp).ok_or("bad \\u escape")?);
                        }
                        other => return Err(format!("bad escape \\{other}")),
                    }
                }
                c => s.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.peek().ok_or("bad \\u")?;
            self.i += 1;
            v = v * 16 + c.to_digit(16).ok_or("bad hex digit")?;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E') {
                self.i += 1;
            } else {
                break;
            }
        }
        let s: String = self.c[start..self.i].iter().collect();
        s.parse::<f64>()
            .map(Json::Num)
            .map_err(|e| format!("bad number {s:?}: {e}"))
    }
}
