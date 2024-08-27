use std::{fs::{self, File}, io::{Read, Write}};

use ab_glyph::FontVec;
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use rand::prelude::*;

fn main() {
    let mut rng = rand::thread_rng();

    let file = File::create("text.zip").unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);

    let (min_len, max_len) = (7, 7);
    let mut text_buffer = String::new();

    let mut text_renderer = TextRenderer::new();
    
    let n = 2usize.pow(5);
    let bar = indicatif::ProgressBar::new((max_len * n) as u64);
    for len in min_len..=max_len {
        let mut texts = String::new();
        for _ in 0..n {
            random_text(
                &mut text_buffer,
                len,
                &mut rng
            );
            texts += &text_buffer;
            texts.push('\n');

            bar.inc(1);
        }
        zip.start_file_from_path(format!("data/texts/l{len}/texts.txt"), options).unwrap();
        zip.write(texts.as_bytes()).unwrap();
    }

    bar.finish();
    zip.finish().unwrap();
}

fn random_text(
    buffer: &mut String,
    len: usize,
    rng: &mut rand::rngs::ThreadRng,
) {
    buffer.clear();
    const CHARS: [char; 39] = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', '='
    ];
    for _ in 0..len {
        let i = rng.gen_range(0..CHARS.len());
        buffer.push(CHARS[i]);
    }
}

struct TextRenderer {
    fonts: Vec<FontVec>,
}
impl TextRenderer {
    fn new() -> Self {
        let mut fonts = Vec::new();
        let dir_paths = [
            "fonts/print",
            "fonts/diverse",
        ];
        for dir_path in dir_paths {
            let dir = fs::read_dir(dir_path).unwrap();
            for entry in dir {
                let path = entry.unwrap().path();
                let buffer = read_file_to_vec(path);
                let font_vec = FontVec::try_from_vec(buffer).unwrap();
                fonts.push(font_vec);
            }
        }
        Self {
            fonts
        }
    }

    fn render(&mut self, buffer: &mut RgbImage) {
        let mut image = RgbImage::new(200, 200);
        // clear
        let background_color = Rgb([255u8, 255u8, 255u8]);
        let rect = imageproc::rect::Rect::at(0, 0).of_size(buffer.width(), buffer.height());
        draw_filled_rect_mut(&mut image, rect, background_color);
    }
}

fn read_file_to_vec<P: AsRef<std::path::Path>>(path: P) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    buffer
}