use std::{fs::{self, File}, io::{Cursor, Read, Write}};

use ab_glyph::FontVec;
use clap::Parser;
use image::{ImageFormat, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut, text_size};
use rand::prelude::*;

#[derive(clap::Parser)]
struct Args {
    #[arg(short, long)]
    config: String,
}

#[derive(serde::Deserialize)]
struct Config {
    number_per_length: usize,
    minl: usize,
    maxl: usize,
    font_dirs: Vec<String>,
    width: u32,
    height: u32,
}

fn main() {
    let args = Args::parse();
    let config = read_file_to_string(args.config);
    let config: Config = toml::from_str(&config).unwrap();

    let mut rng = rand::thread_rng();

    let file = File::create("text.zip").unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);

    let (min_len, max_len) = (config.minl, config.maxl);
    let mut text_buffer = String::new();

    let mut text_renderer = TextRenderer::new(&config.font_dirs);
    let mut image_buffer = RgbImage::new(config.width, config.height);
    let mut encode_buffer: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    
    let n = config.number_per_length;
    let bar = indicatif::ProgressBar::new(((max_len - min_len + 1) * n) as u64);
    for len in min_len..=max_len {
        let mut texts = String::new();
        for i in 0..n {
            random_text(
                &mut text_buffer,
                len,
                &mut rng
            );
            texts += &text_buffer;
            texts.push('\n');

            text_renderer.render(&mut image_buffer, &text_buffer, &mut rng);

            encode_buffer.get_mut().clear();
            encode_buffer.set_position(0);
            image_buffer.write_to(&mut encode_buffer, ImageFormat::Png).unwrap();
            zip.start_file_from_path(format!("data/texts/l{len}/{i}.png"), options).unwrap();
            zip.write_all(encode_buffer.get_ref()).unwrap();

            bar.inc(1);
        }
        zip.start_file_from_path(format!("data/texts/l{len}/texts.txt"), options).unwrap();
        zip.write_all(texts.as_bytes()).unwrap();
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
    const CHARS: [char; 71] = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', '=', '(', ')', '[', ']', '<', '>',
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
    fn new(dir_paths: &[String]) -> Self {
        let mut fonts = Vec::new();
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

    fn render(
        &mut self,
        buffer: &mut RgbImage,
        text: &str,
        rng: &mut rand::rngs::ThreadRng,
    ) {
        let wf = buffer.width() as f32;
        let hf = buffer.height() as f32;

        // clear
        let background_color = Rgb([255u8, 255u8, 255u8]);
        let rect = imageproc::rect::Rect::at(0, 0).of_size(buffer.width(), buffer.height());
        draw_filled_rect_mut(buffer, rect, background_color);

        // random attr
        let font = &self.fonts[rng.gen_range(0..self.fonts.len())];
        let test_scale = 64.0;
        let tested_size = text_size(test_scale, font, text);
        let (twf, thf) = (tested_size.0 as f32, tested_size.1 as f32);

        let max_scale = hf.min(test_scale / twf * wf).min(test_scale / thf * hf);
        let min_scale = 32.0f32.min(max_scale);

        let scale = rng.gen_range(min_scale..=max_scale);
        let width = scale / test_scale * twf;
        let height = scale / test_scale * thf;

        let x = rng.gen_range(0 ..= buffer.width() - width as u32) as i32;
        let y = rng.gen_range(0 ..= buffer.height() - height as u32) as i32;

        // draw
        let text_color = Rgb([0u8, 0u8, 0u8]);
        draw_text_mut(buffer, text_color, x, y, scale, font, text);
    }
}

fn read_file_to_vec<P: AsRef<std::path::Path>>(path: P) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    buffer
}

fn read_file_to_string<P: AsRef<std::path::Path>>(path: P) -> String {
    let mut file = File::open(path).unwrap();
    let mut string = String::new();
    file.read_to_string(&mut string).unwrap();
    string
}