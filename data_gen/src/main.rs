use std::{
    collections::HashMap, fs::{self, File}, io::{Read, Write}, process
};

use rand::prelude::*;
use resvg::usvg;
use clap::Parser;
use tiny_skia::Pixmap;

#[derive(clap::Parser)]
struct Args {
    #[arg(short, long)]
    config: String,
}

#[derive(serde::Deserialize)]
struct Config {
    num_at_depth: HashMap<String, usize>,
    min_num: usize,
    max_length: usize,
}

fn main() {
    let args = Args::parse();
    let config = read_file_to_string(args.config);
    let config: Config = toml::from_str(&config).unwrap();
    
    let mut rng = thread_rng();

    // 1.
    println!("Generating Nodes...");
    let mut nodes = Vec::new();
    for (max_depth_s, num) in config.num_at_depth {
        let max_depth = max_depth_s.parse::<usize>().unwrap();
        for _ in 0..num {
            nodes.push(MathNode::random(&mut rng, 0, max_depth))
        }
    }

    // 2.
    println!("Generating Latex and Text...");
    // Vec<(latex, text)>
    let mut text_map: HashMap<usize, Vec<(String, String)>> = HashMap::new();
    for node in &nodes {
        let latex = node.to_latex();
        let text = node.to_text();
        if text.len() > config.max_length {
            continue; // discarding those longer than the context window of the model
        }
        if let Some(v) = text_map.get_mut(&text.len()) {
            v.push((latex, text));
        } else {
            text_map.insert(text.len(), vec![(latex, text)]);
        }
    }

    drop(nodes);

    // leaving only those length with enough samples
    text_map.retain(|_, v| v.len() > config.min_num);

    // saving latex and text
    for (length, lateses) in &text_map {
        let _ = fs::create_dir(format!("data/latex/l{}", length));
        let mut latex_s = String::new();
        let mut text_s = String::new();
        for (latex, text) in lateses {
            latex_s.push_str(latex);
            latex_s.push('\n');
            text_s.push_str(text);
            text_s.push('\n');
        }
        let mut file = File::create(format!("data/latex/l{}/latex_s.txt", length)).unwrap();
        file.write_all(latex_s.as_bytes()).unwrap();
        let mut file = File::create(format!("data/latex/l{}/text_s.txt", length)).unwrap();
        file.write_all(text_s.as_bytes()).unwrap();
    }

    // 3.
    println!("Generating SVGs...");
    process::Command::new("node")
        .arg("nodejs/tex2svg.js")
        .output()
        .unwrap();
    
    // 4.
    println!("Generating PNGs...");
    let render_options = usvg::Options::default();
    let mut image_buffer = tiny_skia::Pixmap::new(256, 128).unwrap();
    for (length, _) in &text_map {
        let svg_src = read_file_to_string(format!("data/latex/l{}/svg.txt", length));
        svg_src.split("\n").enumerate().for_each(|(i, svg)| {
            let mut png = gen_png(&mut rng, svg, &mut image_buffer, &render_options);
            let mut file = File::create(format!("data/latex/l{}/{}.png", length, i)).unwrap();
            file.write_all(&mut png).unwrap();
        });
    }
}

fn gen_png(
    rng: &mut ThreadRng,
    svg: &str,
    buffer: &mut Pixmap,
    options: &usvg::Options,
) -> Vec<u8> {
    buffer.fill(tiny_skia::Color::WHITE);
    let wf = buffer.width() as f32;
    let hf = buffer.height() as f32;

    let tree = usvg::Tree::from_str(svg, options).unwrap();

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
        &mut buffer.as_mut(),
    );
    buffer.encode_png().unwrap()
}

fn read_file_to_string<P: AsRef<std::path::Path>>(path: P) -> String {
    let mut file = File::open(path).unwrap();
    let mut string = String::new();
    file.read_to_string(&mut string).unwrap();
    string
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
    fn to_text(&self) -> String {
        match self {
            MathNode::Add(node) => node.to_text(),
            MathNode::Sub(node) => node.to_text(),
            MathNode::Uint(num) => num.to_string(),
            MathNode::Neg(node) => node.to_text(),
            MathNode::Val(node) => node.to_text(),
            MathNode::Frac(node) => node.to_text(),
            MathNode::Pow(node) => node.to_text(),
            MathNode::Sqrt(node) => node.to_text(),
            MathNode::Mono(node) => node.to_text(),
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
            let rand_type = rng.gen_range(0..6);
            return if rand_type == 0 {
                MathNode::Add(AddNode::random(rng, depth, max_depth))
            } else if rand_type == 1 {
                MathNode::Sub(SubNode::random(rng, depth, max_depth))
            } else if rand_type == 2 {
                MathNode::Frac(FracNode::random(rng, depth, max_depth))
            } else if rand_type == 3 {
                MathNode::Pow(PowNode::random(rng, depth, max_depth))
            } else if rand_type == 4 {
                MathNode::Sqrt(SqrtNode::random(rng, depth, max_depth))
            } else if rand_type == 5 {
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
    fn to_text(&self) -> String {
        format!("sqrt({})", self.inner.to_text())
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
    fn to_text(&self) -> String {
        let exp_text = match &*self.exp {
            MathNode::Uint(_) | MathNode::Val(_) => self.exp.to_text(),
            _ => format!("({})", self.exp.to_text()),
        };
        format!("{}**{}", self.base.to_text(), exp_text)
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
    fn to_text(&self) -> String {
        let up_text = match &*self.up {
            MathNode::Uint(_) | MathNode::Val(_) => self.up.to_text(),
            _ => format!("({})", self.up.to_text()),
        };
        let down_text = match &*self.down {
            MathNode::Uint(_) | MathNode::Val(_) => self.down.to_text(),
            _ => format!("({})", self.down.to_text()),
        };
        format!("{}/{}", up_text, down_text)
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
    fn to_text(&self) -> String {
        if let Some(sub) = &self.subscription {
            format!("{}_{} ", self.main, sub)
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
        let subscription = if rng.gen::<f32>() < 0.7 {
            None
        } else {
            let sub_len = rng.gen_range(1..=3);
            let mut sub = String::new();
            for _ in 0..sub_len {
                sub.push(SUB_CHARS[rng.gen_range(0..SUB_CHARS.len())]);
            }
            Some(sub)
        };
        Self {
            main: LATTERS[rng.gen_range(0..LATTERS.len())],
            subscription,
        }
    }
}

struct NegNode {
    inner: Box<MathNode>,
}
impl NegNode {
    fn to_latex(&self) -> String {
        match &*self.inner {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => "-(".to_owned() + &self.inner.to_latex() + ")",
            _ => "-".to_owned() + &self.inner.to_latex(),
        }
    }
    fn to_text(&self) -> String {
        match &*self.inner {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => "-(".to_owned() + &self.inner.to_text() + ")",
            _ => "-".to_owned() + &self.inner.to_text(),
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
    fn to_text(&self) -> String {
        let right_text = match &*self.right {
            MathNode::Neg(_) => "(".to_owned() + &self.right.to_text() + ")",
            _ => self.right.to_text(),
        };
        self.left.to_text() + "+" + &right_text
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
    fn to_text(&self) -> String {
        let right_text = match &*self.right {
            MathNode::Sub(_) | MathNode::Add(_) | MathNode::Neg(_) => {
                "(".to_owned() + &self.right.to_text() + ")"
            }
            _ => self.right.to_text(),
        };
        self.left.to_text() + "-" + &right_text
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
