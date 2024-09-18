use rand::prelude::*;
pub enum MathNode {
    Add(AddNode),
    Sub(SubNode),
    Uint(usize),
    Neg(NegNode),
    Val(ValNode),
    Mono(MonoNode),
}
const LATTERS: [char; 50] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    'z',
];
impl MathNode {
    pub fn to_text(&self) -> String {
        match self {
            MathNode::Add(node) => node.to_text(),
            MathNode::Sub(node) => node.to_text(),
            MathNode::Uint(num) => num.to_string(),
            MathNode::Neg(node) => node.to_text(),
            MathNode::Val(node) => node.to_text(),
            MathNode::Mono(node) => node.to_text(),
        }
    }
    pub fn to_show_text(&self) -> String {
        match self {
            MathNode::Add(node) => node.to_show_text(),
            MathNode::Sub(node) => node.to_show_text(),
            MathNode::Uint(num) => num.to_string(),
            MathNode::Neg(node) => node.to_show_text(),
            MathNode::Val(node) => node.to_show_text(),
            MathNode::Mono(node) => node.to_show_text(),
        }
    }
    pub fn random(rng: &mut ThreadRng, mut depth: usize, max_depth: usize) -> Self {
        if depth == max_depth {
            let rand_type = rng.gen_range(0..2);
            return if rand_type == 0 {
                MathNode::Uint(rng.gen_range(0..111))
            } else if rand_type == 1 {
                MathNode::Val(ValNode::random(rng))
            } else {
                unreachable!()
            };
        } else {
            depth += 1;
            let rand_type = rng.gen_range(0..5);
            return if rand_type == 0 {
                MathNode::Add(AddNode::random(rng, depth, max_depth))
            } else if rand_type == 1 {
                MathNode::Sub(SubNode::random(rng, depth, max_depth))
            } else if rand_type == 2 {
                MathNode::Neg(NegNode::random(rng, depth, max_depth))
            } else if rand_type == 3 {
                MathNode::Val(ValNode::random(rng))
            } else if rand_type == 4 {
                MathNode::Mono(MonoNode::random(rng, depth, max_depth))
            } else {
                unreachable!()
            };
        }
    }
    fn random_non_num(rng: &mut ThreadRng, mut depth: usize, max_depth: usize) -> Self {
        if depth == max_depth {
            return MathNode::Val(ValNode::random(rng));
        }
        depth += 1;
        let rand_type = rng.gen_range(0..5);
        return if rand_type == 0 {
            MathNode::Add(AddNode::random(rng, depth, max_depth))
        } else if rand_type == 1 {
            MathNode::Sub(SubNode::random(rng, depth, max_depth))
        } else if rand_type == 2 {
            MathNode::Neg(NegNode::random(rng, depth, max_depth))
        } else if rand_type == 3 {
            MathNode::Val(ValNode::random(rng))
        } else if rand_type == 4 {
            MathNode::Mono(MonoNode::random(rng, depth, max_depth))
        } else {
            unreachable!()
        };
    }
    pub fn random_single_char(rng: &mut ThreadRng) -> Self {
        let rand_type = rng.gen_range(0..=1);
        if rand_type == 0 {
            let num = rng.gen_range(0..10);
            Self::Uint(num)
        } else {
            let char = LATTERS[rng.gen_range(0..LATTERS.len())];
            Self::Val(ValNode { main: char })
        }
    }
}
pub struct ValNode {
    main: char,
}
impl ValNode {
    fn to_text(&self) -> String {
        self.main.to_string()
    }
    fn to_show_text(&self) -> String {
        self.to_text()
    }
    fn random(rng: &mut ThreadRng) -> Self {
        // removed o and O for they are too similar to 0
        Self {
            main: LATTERS[rng.gen_range(0..LATTERS.len())],
        }
    }
}

pub struct NegNode {
    inner: Box<MathNode>,
}
impl NegNode {
    fn to_text(&self) -> String {
        match &*self.inner {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => "-(".to_owned() + &self.inner.to_text() + ")",
            _ => "-".to_owned() + &self.inner.to_text(),
        }
    }
    fn to_show_text(&self) -> String {
        match &*self.inner {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => "-(".to_owned() + &self.inner.to_show_text() + ")",
            _ => "-".to_owned() + &self.inner.to_show_text(),
        }
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            inner: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

pub struct AddNode {
    left: Box<MathNode>,
    right: Box<MathNode>,
}
impl AddNode {
    fn to_text(&self) -> String {
        let right_text = match &*self.right {
            MathNode::Neg(_) => "(".to_owned() + &self.right.to_text() + ")",
            _ => self.right.to_text(),
        };
        self.left.to_text() + "+" + &right_text
    }
    fn to_show_text(&self) -> String {
        let right_text = match &*self.right {
            MathNode::Neg(_) => "(".to_owned() + &self.right.to_show_text() + ")",
            _ => self.right.to_show_text(),
        };
        self.left.to_show_text() + "+" + &right_text
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            left: Box::new(MathNode::random(rng, depth, max_depth)),
            right: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

pub struct SubNode {
    left: Box<MathNode>,
    right: Box<MathNode>,
}
impl SubNode {
    fn to_text(&self) -> String {
        let right_text = match &*self.right {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => {
                "(".to_owned() + &self.right.to_text() + ")"
            }
            _ => self.right.to_text(),
        };
        self.left.to_text() + "-" + &right_text
    }
    fn to_show_text(&self) -> String {
        let right_text = match &*self.right {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => {
                "(".to_owned() + &self.right.to_show_text() + ")"
            }
            _ => self.right.to_show_text(),
        };
        self.left.to_show_text() + "-" + &right_text
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            left: Box::new(MathNode::random(rng, depth, max_depth)),
            right: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

pub struct MonoNode {
    coef: usize,
    vals: Vec<MathNode>,
}
impl MonoNode {
    fn to_text(&self) -> String {
        let mut result = self.coef.to_string();
        for val in &self.vals {
            let val_text = match val {
                MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) | MathNode::Mono(_) => {
                    format!("*({})", val.to_text())
                }
                _ => "*".to_owned() + &val.to_text(),
            };
            result.push_str(&val_text);
        }
        result
    }
    fn to_show_text(&self) -> String {
        let mut result = self.coef.to_string();
        for val in &self.vals {
            let val_text = match val {
                MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) | MathNode::Mono(_) => {
                    format!("({})", val.to_show_text())
                }
                _ => val.to_show_text(),
            };
            result.push_str(&val_text);
        }
        result
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        let coef = rng.gen_range(2..111);
        let val_len = rng.gen_range(1..=3);
        let mut vals = Vec::with_capacity(val_len);
        for _ in 0..val_len {
            vals.push(MathNode::random_non_num(rng, depth, max_depth));
        }
        Self { coef, vals }
    }
}