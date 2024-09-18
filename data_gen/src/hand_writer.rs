use std::collections::HashMap;
use tiny_skia::{Pixmap, PixmapPaint, Transform};
use rand::prelude::*;
use std::fs;

pub struct HandWriter {
    char_imgs: HashMap<char, Vec<Pixmap>>
}
impl HandWriter {
    pub fn new() -> Self {
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
    pub fn rand_draw(
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