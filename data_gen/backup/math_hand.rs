use std::{collections::HashMap, fs::{self, File}, io::{Read, Write}};
use clap::Parser;
use tiny_skia::{Color, Pixmap, PixmapPaint, Transform};
use rand::prelude::*;

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
    
    // 0.
    println!("initiating...");
    let mut rng = thread_rng();
    let mut canvas = Pixmap::new(256, 128).unwrap();
    let hand_writer = HandWriter::new();

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
    println!("Generating Show Text and Text...");
    // Vec<(show_text, text)>
    let mut text_map: HashMap<usize, Vec<(String, String)>> = HashMap::new();
    for node in &nodes {
        let show_text = node.to_show_text();
        let text = node.to_text();
        if text.len() > config.max_length {
            continue; // discarding those longer than the context window of the model
        }
        if let Some(v) = text_map.get_mut(&text.len()) {
            v.push((show_text, text));
        } else {
            text_map.insert(text.len(), vec![(show_text, text)]);
        }
    }
    drop(nodes);

    // leaving only those length with enough samples
    text_map.retain(|_, v| v.len() > config.min_num);

    // saving text
    for (length, texts) in &text_map {
        let _ = fs::create_dir(format!("data/hand_math/l{}", length));
        let mut text_s = String::new();
        for (_, text) in texts {
            text_s.push_str(text);
            text_s.push('\n');
        }
        let mut file = File::create(format!("data/hand_math/l{}/text.txt", length)).unwrap();
        file.write_all(text_s.as_bytes()).unwrap();
    }

    // 3.
    println!("Generating images...");
    for (length, texts) in &text_map {
        for i in 0..texts.len() {
            let (show_text, _) = &texts[i];
            canvas.fill(Color::WHITE);
            hand_writer.rand_draw(show_text, &mut rng, &mut canvas);
            canvas.save_png(format!("data/hand_math/l{}/{}.png", length, i)).unwrap();
        }
    }
}

fn read_file_to_string<P: AsRef<std::path::Path>>(path: P) -> String {
    let mut file = File::open(path).unwrap();
    let mut string = String::new();
    file.read_to_string(&mut string).unwrap();
    string
}


struct HandWriter {
    char_imgs: HashMap<char, Vec<Pixmap>>
}
impl HandWriter {
    fn new() -> Self {
        const CHARS: [char; 66] = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', '(', ')'
        ];

        let mut char_imgs = HashMap::new();

        for char in CHARS {
            let mut v = Vec::new();
            for entry in fs::read_dir(format!("data/chars/{}", char)).unwrap() {
                if entry.as_ref().unwrap().file_type().unwrap().is_dir() {
                    continue;
                }
                let path = entry.unwrap().path();
                v.push(Pixmap::load_png(path).unwrap());
            }
            char_imgs.insert(char, v);
        };
        Self {
            char_imgs
        }
    }
    fn rand_draw(
        &self,
        text: &str,
        rng: &mut ThreadRng,
        canvas: &mut Pixmap,
    ) {
        // first char at (0, 0)
        // +y
        // |
        // |____+x
        let mut chosen_imgs: Vec<&Pixmap> = Vec::new();
        let mut layouts: Vec<CharLayout> = Vec::new();
        let mut total_width = 0.0;
        let mut total_height: f32 = 0.0;

        let char_size = rng.gen_range(20.0..100.0);
        for c in text.chars() {
            const OTHER_BIG_CHARS: [char; 11] = [
                'b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'p', 'q', 'y',
            ];
            const BIGGER_CHARS: [char; 2] = [
                '(', ')'
            ];
            let h;
            if c.is_ascii_uppercase() || c.is_ascii_digit() || OTHER_BIG_CHARS.contains(&c) {
                h = char_size;
            } else if BIGGER_CHARS.contains(&c) {
                h = char_size * 1.3;
            } else {
                h = char_size * 0.7;
            };

            const LOWER_CHARS: [char; 5] = [
                'g', 'j', 'p', 'q', 'y',
            ];
            let y = if LOWER_CHARS.contains(&c) {
                -char_size * 0.34
            } else if c == '(' || c == ')' {
                -char_size * 0.15
            } else {
                0.0
            };

            let char_img_vec = self.char_imgs.get(&c).unwrap();
            let char_img = &char_img_vec[rng.gen_range(0..char_img_vec.len())];
            let w = h / char_img.height() as f32 * char_img.width() as f32;
            chosen_imgs.push(char_img);

            let layout = CharLayout {
                x: total_width,
                y,
                w,
                h
            };
            let random_gap = rng.gen_range(0.0..char_size * 0.2);
            total_width += layout.w + random_gap;
            total_height = total_height.max(layout.h);
            layouts.push(layout);
        }

        // coord transform
        for layout in &mut layouts {
            layout.y = total_height - layout.h - layout.y;
        }
        
        // scale
        let mut scale = 1.0;
        let (cwf, chf) = (canvas.width() as f32, canvas.height() as f32);
        if total_width > cwf {
            scale = cwf / total_width;
        }
        if total_height > chf {
            scale = scale.min(chf / total_height);
        }
        for layout in &mut layouts {
            layout.x *= scale;
            layout.y *= scale;
            layout.w *= scale;
            layout.h *= scale;
        }

        // translate
        let offset_x = rng.gen_range(0.0..=(cwf - total_width * scale).max(0.0));
        let offset_y = rng.gen_range(0.0..=(chf - total_height * scale).max(0.0));
        for layout in &mut layouts {
            layout.x += offset_x;
            layout.y += offset_y;
        }

        for i in 0..chosen_imgs.len() {
            let layout = &layouts[i];

            let char_img = chosen_imgs[i];

            canvas.draw_pixmap(
                0,
                0,
                char_img.as_ref(),
                &PixmapPaint::default(),
                Transform::from_scale(layout.w / char_img.width() as f32, layout.h / char_img.height() as f32)
                .post_translate(layout.x, layout.y),
                None,
            );
        }
    }
}
#[derive(Debug)]
struct CharLayout {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

enum MathNode {
    Add(AddNode),
    Sub(SubNode),
    Uint(usize),
    Neg(NegNode),
    Val(ValNode),
    Mono(MonoNode),
}
impl MathNode {
    fn to_text(&self) -> String {
        match self {
            MathNode::Add(node) => node.to_text(),
            MathNode::Sub(node) => node.to_text(),
            MathNode::Uint(num) => num.to_string(),
            MathNode::Neg(node) => node.to_text(),
            MathNode::Val(node) => node.to_text(),
            MathNode::Mono(node) => node.to_text(),
        }
    }
    fn to_show_text(&self) -> String {
        match self {
            MathNode::Add(node) => node.to_show_text(),
            MathNode::Sub(node) => node.to_show_text(),
            MathNode::Uint(num) => num.to_string(),
            MathNode::Neg(node) => node.to_show_text(),
            MathNode::Val(node) => node.to_show_text(),
            MathNode::Mono(node) => node.to_show_text(),
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
            let rand_type = rng.gen_range(0..3);
            return if rand_type == 0 {
                MathNode::Add(AddNode::random(rng, depth, max_depth))
            } else if rand_type == 1 {
                MathNode::Sub(SubNode::random(rng, depth, max_depth))
            } else if rand_type == 2 {
                MathNode::Mono(MonoNode::random(rng, depth, max_depth))
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
}
struct ValNode {
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
        const LATTERS: [char; 52] = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z',
        ];
        Self {
            main: LATTERS[rng.gen_range(0..LATTERS.len())],
        }
    }
}

struct NegNode {
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

struct AddNode {
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

struct SubNode {
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

struct MonoNode {
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