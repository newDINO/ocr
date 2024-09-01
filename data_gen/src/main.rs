use std::{
    fs,
    io::{Read, Write},
    process,
};

use indicatif::ProgressBar;
use rand::prelude::*;
use resvg::usvg;

fn main() {
    let mut rng = thread_rng();

    let n = 16;

    let mut nodes = Vec::with_capacity(n);
    println!("Generating Nodes:");
    let bar = ProgressBar::new(n as u64);
    for _ in 0..n {
        bar.inc(1);
        nodes.push(MathNode::random(&mut rng, 0, 2))
    }
    bar.finish();

    let mut lateses = Vec::with_capacity(n);
    println!("Generating Latex:");
    let bar = ProgressBar::new(n as u64);
    for i in 0..n {
        bar.inc(1);
        lateses.push(nodes[i].to_latex());
    }
    bar.finish();
    drop(nodes);
    let mut texts = String::new();
    for latex in &lateses {
        texts.push_str(latex);
        texts.push('\n');
    }
    let mut file = fs::File::create("data/latex/texts.txt").unwrap();
    file.write_all(texts.as_bytes()).unwrap();

    println!("Generating Svgs:");
    process::Command::new("node")
        .arg("nodejs/tex2svg.js")
        .output()
        .unwrap();
    let svg_texts = read_file_to_string("data/latex/svg.txt");
    let svgs = svg_texts.split('\n').collect::<Vec<&str>>();

    let mut pngs = Vec::with_capacity(n);
    println!("Generating Pngs:");
    let mut image_buffer = tiny_skia::Pixmap::new(256, 128).unwrap();
    let wf = image_buffer.width() as f32;
    let hf = image_buffer.height() as f32;
    let reander_option = usvg::Options::default();
    let bar = ProgressBar::new(n as u64);
    for i in 0..n {
        bar.inc(1);
        image_buffer.fill(tiny_skia::Color::WHITE);

        let tree = usvg::Tree::from_str(svgs[i], &reander_option).unwrap();

        let original_size = tree.size();
        let max_scale = (wf / original_size.width()).min(hf / original_size.height());
        let min_scale = 2.0f32.min(max_scale);
        let scale = rng.gen_range(min_scale..=max_scale);
        let width = original_size.width() * scale;
        let height = original_size.height() * scale;

        let x = rng.gen_range(0.0..=wf - width);
        let y = rng.gen_range(0.0..=hf - height);

        resvg::render(
            &tree,
            usvg::Transform::from_scale(scale, scale).post_translate(x, y),
            &mut image_buffer.as_mut(),
        );
        pngs.push(image_buffer.encode_png().unwrap());
    }
    bar.finish();
    drop(svgs);

    println!("Saving Pngs:");
    let bar = ProgressBar::new(n as u64);
    for i in 0..n {
        bar.inc(1);
        let mut file = fs::File::create(format!("data/latex/{i}.png")).unwrap();
        file.write_all(&pngs[i]).unwrap();
    }
    bar.finish();
}

fn read_file_to_string(path: &str) -> String {
    let mut file = fs::File::open(path).unwrap();
    let mut result = String::new();
    file.read_to_string(&mut result).unwrap();
    result
}

enum MathNode {
    Add(AddNode),
    Sub(SubNode),
    Uint(usize),
    Neg(NegNode),
    Val(ValNode),
    Frac(FracNode),
    Pow(PowNode),
    Sqrt(SqrtNode),
    Mono(MonoNode),
}
impl MathNode {
    fn to_latex(&self) -> String {
        match self {
            MathNode::Add(node) => node.to_latex(),
            MathNode::Sub(node) => node.to_latex(),
            MathNode::Uint(num) => num.to_string(),
            MathNode::Neg(node) => node.to_latex(),
            MathNode::Val(node) => node.to_latex(),
            MathNode::Frac(node) => node.to_latex(),
            MathNode::Pow(node) => node.to_latex(),
            MathNode::Sqrt(node) => node.to_latex(),
            MathNode::Mono(node) => node.to_latex(),
        }
    }
    fn random(rng: &mut ThreadRng, mut depth: usize, max_depth: usize) -> Self {
        if depth == max_depth {
            let rand_type = rng.gen_range(0..2);
            return if rand_type == 0 {
                MathNode::Uint(rng.gen_range(0..111))
            } else if rand_type == 1 {
                MathNode::Val(ValNode::random(rng))
            } else {
                unreachable!()
            };
        } else if depth == 0 {
            depth += 1;
            let rand_type = rng.gen_range(0..7);
            return if rand_type == 0 {
                MathNode::Add(AddNode::random(rng, depth, max_depth))
            } else if rand_type == 1 {
                MathNode::Sub(SubNode::random(rng, depth, max_depth))
            } else if rand_type == 2 {
                MathNode::Val(ValNode::random(rng))
            } else if rand_type == 3 {
                MathNode::Frac(FracNode::random(rng, depth, max_depth))
            } else if rand_type == 4 {
                MathNode::Pow(PowNode::random(rng, depth, max_depth))
            } else if rand_type == 5 {
                MathNode::Sqrt(SqrtNode::random(rng, depth, max_depth))
            } else if rand_type == 6 {
                MathNode::Mono(MonoNode::random(rng, depth, max_depth))
            } else {
                unreachable!()
            };
        } else {
            depth += 1;
            let rand_type = rng.gen_range(0..8);
            return if rand_type == 0 {
                MathNode::Add(AddNode::random(rng, depth, max_depth))
            } else if rand_type == 1 {
                MathNode::Sub(SubNode::random(rng, depth, max_depth))
            } else if rand_type == 2 {
                MathNode::Neg(NegNode::random(rng, depth, max_depth))
            } else if rand_type == 3 {
                MathNode::Val(ValNode::random(rng))
            } else if rand_type == 4 {
                MathNode::Frac(FracNode::random(rng, depth, max_depth))
            } else if rand_type == 5 {
                MathNode::Pow(PowNode::random(rng, depth, max_depth))
            } else if rand_type == 6 {
                MathNode::Sqrt(SqrtNode::random(rng, depth, max_depth))
            } else if rand_type == 7 {
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
        let rand_type = rng.gen_range(0..8);
        return if rand_type == 0 {
            MathNode::Add(AddNode::random(rng, depth, max_depth))
        } else if rand_type == 1 {
            MathNode::Sub(SubNode::random(rng, depth, max_depth))
        } else if rand_type == 2 {
            MathNode::Neg(NegNode::random(rng, depth, max_depth))
        } else if rand_type == 3 {
            MathNode::Val(ValNode::random(rng))
        } else if rand_type == 4 {
            MathNode::Frac(FracNode::random(rng, depth, max_depth))
        } else if rand_type == 5 {
            MathNode::Pow(PowNode::random(rng, depth, max_depth))
        } else if rand_type == 6 {
            MathNode::Sqrt(SqrtNode::random(rng, depth, max_depth))
        } else if rand_type == 7 {
            MathNode::Mono(MonoNode::random(rng, depth, max_depth))
        } else {
            unreachable!()
        };
    }
}

struct SqrtNode {
    inner: Box<MathNode>,
}
impl SqrtNode {
    fn to_latex(&self) -> String {
        format!("\\sqrt{{{}}}", self.inner.to_latex())
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            inner: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

struct PowNode {
    base: Box<MathNode>,
    exp: Box<MathNode>,
}
impl PowNode {
    fn to_latex(&self) -> String {
        let base_string = match &*self.base {
            MathNode::Add(_)
            | MathNode::Sub(_)
            | MathNode::Neg(_)
            | MathNode::Frac(_)
            | MathNode::Pow(_)
            | MathNode::Mono(_) => {
                format!("({})", self.base.to_latex())
            }
            _ => self.base.to_latex(),
        };
        format!("{}^{{{}}}", base_string, self.exp.to_latex())
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            base: Box::new(MathNode::random(rng, depth, max_depth)),
            exp: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

struct FracNode {
    up: Box<MathNode>,
    down: Box<MathNode>,
}
impl FracNode {
    fn to_latex(&self) -> String {
        format!(
            "\\frac{{{}}}{{{}}}",
            self.up.to_latex(),
            self.down.to_latex()
        )
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            up: Box::new(MathNode::random(rng, depth, max_depth)),
            down: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

struct ValNode {
    main: char,
    subscription: Option<String>,
}
impl ValNode {
    fn to_latex(&self) -> String {
        if let Some(sub) = &self.subscription {
            if sub.len() > 1 {
                self.main.to_string() + "_{" + sub + "}"
            } else {
                self.main.to_string() + "_" + sub
            }
        } else {
            self.main.to_string()
        }
    }
    fn random(rng: &mut ThreadRng) -> Self {
        const LATTERS: [char; 52] = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z',
        ];
        const SUB_CHARS: [char; 36] = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '0',
        ];
        let sub_len = rng.gen_range(0..=3);
        if sub_len == 0 {
            Self {
                main: LATTERS[rng.gen_range(0..LATTERS.len())],
                subscription: None,
            }
        } else {
            let mut sub = String::new();
            for _ in 0..sub_len {
                sub.push(SUB_CHARS[rng.gen_range(0..SUB_CHARS.len())]);
            }
            Self {
                main: LATTERS[rng.gen_range(0..LATTERS.len())],
                subscription: Some(sub),
            }
        }
    }
}

struct NegNode {
    inner: Box<MathNode>,
}
impl NegNode {
    fn to_latex(&self) -> String {
        match &*self.inner {
            MathNode::Sub(_) | MathNode::Add(_) => "-(".to_owned() + &self.inner.to_latex() + ")",
            _ => "-".to_owned() + &self.inner.to_latex(),
        }
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            inner: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

struct AddNode {
    left: Box<MathNode>,
    right: Box<MathNode>,
}
impl AddNode {
    fn to_latex(&self) -> String {
        let right_latex = match &*self.right {
            MathNode::Neg(_) => "(".to_owned() + &self.right.to_latex() + ")",
            _ => self.right.to_latex(),
        };
        self.left.to_latex() + "+" + &right_latex
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            left: Box::new(MathNode::random(rng, depth, max_depth)),
            right: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

struct SubNode {
    left: Box<MathNode>,
    right: Box<MathNode>,
}
impl SubNode {
    fn to_latex(&self) -> String {
        let right_latex = match &*self.right {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => {
                "(".to_owned() + &self.right.to_latex() + ")"
            }
            _ => self.right.to_latex(),
        };
        self.left.to_latex() + "-" + &right_latex
    }
    fn random(rng: &mut ThreadRng, depth: usize, max_depth: usize) -> Self {
        Self {
            left: Box::new(MathNode::random(rng, depth, max_depth)),
            right: Box::new(MathNode::random(rng, depth, max_depth)),
        }
    }
}

struct MonoNode {
    coef: usize,
    vals: Vec<MathNode>,
}
impl MonoNode {
    fn to_latex(&self) -> String {
        let mut result = self.coef.to_string();
        for val in &self.vals {
            let val_latex = match val {
                MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) | MathNode::Mono(_) => {
                    format!("({})", val.to_latex())
                }
                _ => val.to_latex(),
            };
            result.push_str(&val_latex);
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
