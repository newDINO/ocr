use std::{fs::File, io::Write};

use rand::prelude::*;
fn main() {
    let mut text_drawer = TextDrawer::new();
    let mut rng = rand::thread_rng();

    let (min_len, max_len) = (1, 16);
    let mut text_buffer = Vec::with_capacity(max_len);

    let file = File::create("text_math.zip").unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);
    
    let n = 2usize.pow(13);
    let bar = indicatif::ProgressBar::new((max_len * n) as u64);
    for len in min_len..max_len {
        let mut texts = String::new();
        for i in 0..n {
            random_text(
                &mut text_buffer,
                len,
                &mut rng
            );
            let text = std::str::from_utf8(&text_buffer).unwrap();
            texts += text;
            texts.push('\n');
            let img = text_drawer.random_draw(&text, &mut rng, i);

            zip.start_file_from_path(format!("data/texts/l{len}/{i}.png"), options).unwrap();
            zip.write(&img).unwrap();

            bar.inc(1);
        }
        zip.start_file_from_path(format!("data/texts/l{len}/texts.txt"), options).unwrap();
        zip.write(texts.as_bytes()).unwrap();
    }
    
    bar.finish();
    zip.finish().unwrap();
}

fn random_text<'a>(
    buffer: &'a mut Vec<u8>,
    len: usize,
    rng: &mut rand::rngs::ThreadRng,
) {
    buffer.clear();
    const CHARS: [char; 41] = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', '=', '(', ')'
    ];
    for _ in 0..len {
        let i = rng.gen_range(0..CHARS.len());
        buffer.push(CHARS[i] as u8);
    }
}

struct TextDrawer {
    font_system: cosmic_text::FontSystem,
    swash_cache: cosmic_text::SwashCache,
    buffer: cosmic_text::Buffer,
    pixmap: tiny_skia::Pixmap,
}

impl TextDrawer {
    fn new() -> Self {
        let mut database = fontdb::Database::new();
        database.load_fonts_dir("fonts");
        let mut font_system = cosmic_text::FontSystem::new_with_locale_and_db("".to_owned(), database);
        // let mut font_system = cosmic_text::FontSystem::new();
        let swash_cache = cosmic_text::SwashCache::new();
        let metrics = cosmic_text::Metrics::new(14.0, 20.0);
        let buffer = cosmic_text::Buffer::new(&mut font_system, metrics);
        let pixmap = tiny_skia::Pixmap::new(256, 128).unwrap();
        Self {
            font_system, swash_cache, buffer, pixmap
        }
    }
    fn draw(
        &mut self,
        text: &str,
        x: u32,
        y: u32,
        color: cosmic_text::Color,
        size: f32,
        attrs: cosmic_text::Attrs,
    ) -> Vec<u8> {
        let mut buffer = self.buffer.borrow_with(&mut self.font_system);
        buffer.set_text(text, attrs, cosmic_text::Shaping::Advanced);
        buffer.set_metrics(cosmic_text::Metrics::new(size, size));

        self.pixmap.fill(tiny_skia::Color::WHITE);
        buffer.draw(&mut self.swash_cache, color, |tx, ty, _, _, color| {
            let fx = x + tx as u32;
            let fy = y + ty as u32;
            draw_pixel(&mut self.pixmap, fx, fy, color.as_rgba());
        });

        self.pixmap.encode_png().unwrap()
    }
    fn text_width(
        &mut self,
        text: &str,
        size: f32,
        attrs: cosmic_text::Attrs,
    ) -> f32 {
        let mut buffer = self.buffer.borrow_with(&mut self.font_system);
        buffer.set_text(text, attrs, cosmic_text::Shaping::Advanced);
        buffer.set_metrics(cosmic_text::Metrics::new(size, size));

        let mut x = 0.0;
        let mut w = 0.0;
        for run in buffer.layout_runs() {
            for glyph in run.glyphs.iter() {
                x = glyph.x;
                w = glyph.w;
            }
        }
        x + w
    }
    fn random_pos(&mut self, text: &str, attrs: cosmic_text::Attrs, rng: &mut rand::rngs::ThreadRng) -> (u32, u32, f32) {
        let wf = self.pixmap.width() as f32;
        let hf = self.pixmap.height() as f32;
        let test_size = 16.0;
        let test_width = self.text_width(text, test_size, attrs);

        let max_size = hf.min(test_size / test_width * wf);
        let min_size = 20.0f32.min(max_size);
        let size = rng.gen_range(min_size..=max_size);
        let width = test_width / test_size * size;

        let x = rng.gen_range(0..=self.pixmap.width().saturating_sub(width as u32));
        let y = rng.gen_range(0..=self.pixmap.height().saturating_sub(size as u32));
        (x, y, size)
    }
    fn random_attrs(&mut self, i: usize) -> cosmic_text::Attrs<'static> {
        let mut result = cosmic_text::Attrs::new();

        use cosmic_text::Family;
        const FAMILIES: [cosmic_text::Family<'static>; 3] = [
            Family::Monospace,
            Family::Serif,
            Family::Name("DejaVu Sans")
        ];
        let family = FAMILIES[i / 2 % FAMILIES.len()];
        result.family = family;

        use cosmic_text::Stretch;
        const STRETCHES: [Stretch; 2] = [
            Stretch::Condensed,
            Stretch::Normal,
        ];
        result.stretch = STRETCHES[i % STRETCHES.len()];
        if self.font_system.get_font_matches(result).len() == 0 {
            result.stretch = Stretch::Normal;
        }

        use cosmic_text::Weight;
        const WEIGHTS: [Weight; 3] = [
            Weight::BOLD,
            Weight::NORMAL,
            Weight::LIGHT
        ];
        result.weight = WEIGHTS[i % WEIGHTS.len()];
        if self.font_system.get_font_matches(result).len() == 0 {
            result.weight = Weight::NORMAL;
        }

        use cosmic_text::Style;
        const STYLES: [Style; 2] = [
            Style::Normal,
            Style::Oblique
        ];
        result.style = STYLES[i % 2];
        if self.font_system.get_font_matches(result).len() == 0 {
            result.style = Style::Normal
        }

        result
    }
    fn random_draw(&mut self, text: &str, rng: &mut rand::rngs::ThreadRng, index: usize) -> Vec<u8> {
        let attrs = self.random_attrs(index);
        let (x, y, size) = self.random_pos(text, attrs, rng);
        let color = cosmic_text::Color::rgba(0, 0, 0, 255);

        self.draw(text, x, y, color, size, attrs)
    }
}

fn draw_pixel(pixmap: &mut tiny_skia::Pixmap, x: u32, y: u32, color: [u8; 4]) {
    let color = tiny_skia::ColorU8::from_rgba(color[0], color[1], color[2], color[3]);
    if x >= pixmap.width() || y >= pixmap.height() {
        return;
    }
    let index = (x + y * pixmap.width()) as usize;
    let prev_color = pixmap.pixels()[index].demultiply();
    pixmap.pixels_mut()[index] = draw_on_color(prev_color, color).premultiply();
}

fn coloru8_to_color(coloru8: tiny_skia::ColorU8) -> tiny_skia::Color {
    tiny_skia::Color::from_rgba8(coloru8.red(), coloru8.green(), coloru8.blue(), coloru8.alpha())
}

fn draw_on_color(canvas_coloru8: tiny_skia::ColorU8, paint_coloru8: tiny_skia::ColorU8) -> tiny_skia::ColorU8 {
    let canvas_color = coloru8_to_color(canvas_coloru8);
    let paint_color = coloru8_to_color(paint_coloru8);
    let canvas_k = 1.0 - paint_color.alpha();
    let paint_k = paint_color.alpha();
    let r = canvas_k * canvas_color.red() + paint_k * paint_color.red();
    let g = canvas_k * canvas_color.green() + paint_k * paint_color.green();
    let b = canvas_k * canvas_color.blue() + paint_k * paint_color.blue();
    let a = canvas_k * canvas_color.alpha() + paint_k;
    let color = tiny_skia::Color::from_rgba(r, g, b, a.min(1.0)).unwrap();
    color.to_color_u8()
}