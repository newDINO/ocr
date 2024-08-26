use std::{fs::File, io::Write};

use mathjax::MathJax;
use rand::prelude::*;

fn main() {
    let mut rng = thread_rng();
    let renderer = MathJax::new().unwrap();
    let mut pixmap = tiny_skia::Pixmap::new(256, 128).unwrap();

    let mut texts = String::new();

    let n = 10;
    let bar = indicatif::ProgressBar::new(n);
    for i in 0..16 {
        let tree = MathNode::random(&mut rng, 0, 3);
        let expression = tree.to_letax();

        texts.push_str(&expression);
        texts.push('\n');

        let result = renderer.render(expression).unwrap();

        let latex_image = result.into_image(rng.gen_range(1.0..10.0)).unwrap();
        let latex_pixmap = image_to_tiny_skia(latex_image);

        pixmap.fill(tiny_skia::Color::WHITE);
        let mut sx = 1.0;
        let mut sy = 1.0;
        let mut x = 0;
        let mut y = 0;
        if latex_pixmap.width() > pixmap.width() {
            sx = pixmap.width() as f32 / latex_pixmap.width() as f32;
            sy = sx;
        } else {
            x = rng.gen_range(0..=(pixmap.width() - latex_pixmap.width()) as i32)
        }
        if latex_pixmap.height() > pixmap.height() {
            sy = pixmap.height() as f32 / latex_pixmap.height() as f32;
            if sy < 0.7 * sx {
                sx = sy;
            }
        } else {
            y = rng.gen_range(0..=(pixmap.height() - latex_pixmap.height()) as i32)
        }
        pixmap.draw_pixmap(
            x,
            y,
            latex_pixmap.as_ref(),
            &tiny_skia::PixmapPaint {
                opacity: 1.0,
                blend_mode: tiny_skia::BlendMode::SourceOver,
                quality: tiny_skia::FilterQuality::Bicubic,
            },
            tiny_skia::Transform::from_scale(sx, sy),
            None,
        );
        pixmap.save_png(format!("data/latex/0/{i}.png")).unwrap();
        bar.inc(1);
    }
    bar.finish();
    let mut file = File::create("data/latex/0/latex.txt").unwrap();
    file.write_all(texts.as_bytes()).unwrap();
}

fn image_to_tiny_skia(image: image::DynamicImage) -> tiny_skia::Pixmap {
    let width = image.width();
    let height = image.height();
    let buffer = image.to_rgba8();
    let size = tiny_skia::IntSize::from_wh(width, height).unwrap();
    tiny_skia::Pixmap::from_vec(buffer.into_vec(), size).unwrap()
}

enum MathNode {
    Num(u64),
    Var(VarNode),
    Add(Box<AddNode>),
    Sub(Box<SubNode>),
}

impl MathNode {
    fn random(rng: &mut rand::rngs::ThreadRng, depth: usize, max_depth: usize) -> Self {
        let rand_num = if depth == max_depth {
            rng.gen_range(0..2)
        } else {
            if rng.gen::<f32>() > 0.7 {
                rng.gen_range(0..4)
            } else {
                rng.gen_range(2..4)
            }
        };
        if rand_num == 0 {
            MathNode::Num(rng.gen_range(0..128))
        } else if rand_num == 1 {
            const CHARS: [char; 26] = [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            ];
            let i = rng.gen_range(0..CHARS.len());
            let c = CHARS[i];
            MathNode::Var(VarNode { main: c })
        } else if rand_num == 2 {
            let left = MathNode::random(rng, depth + 1, max_depth);
            let right = MathNode::random(rng, depth + 1, max_depth);
            let node = AddNode { left, right };
            MathNode::Add(Box::new(node))
        } else if rand_num == 3 {
            let left = MathNode::random(rng, depth + 1, max_depth);
            let right = MathNode::random(rng, depth + 1, max_depth);
            let node = SubNode { left, right };
            MathNode::Sub(Box::new(node))
        } else {
            unreachable!()
        }
    }
    fn to_letax(&self) -> String {
        match self {
            MathNode::Add(node) => node.left.to_letax() + "+" + &node.right.to_letax(),
            MathNode::Sub(node) => {
                match node.right {
                    MathNode::Add(_) | MathNode::Sub(_) => node.left.to_letax() + "-(" + &node.right.to_letax() + ")",
                    _ => node.left.to_letax() + "-" + &node.right.to_letax()
                }
            },
            MathNode::Num(n) => n.to_string(),
            MathNode::Var(var) => var.main.to_string(),
        }
    }
}

struct AddNode {
    left: MathNode,
    right: MathNode,
}

struct SubNode {
    left: MathNode,
    right: MathNode,
}

struct VarNode {
    main: char,
}
