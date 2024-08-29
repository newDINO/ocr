use std::fs;

use image::{imageops, ImageFormat};

fn main() {
    const CHARS: [char; 26] = [
        // 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ];
    for i in 0..26 {
        let _ = fs::create_dir(format!("data/chars/{}", CHARS[i]));
        let dir = fs::read_dir(format!("data/original_chars/Sample0{:02}", i + 37)).unwrap();
        for (j, entry) in dir.enumerate() {
            let path = entry.unwrap().path();
            let image = image::open(path).unwrap().into_rgb8();

            let cropped = crop_to_fit(&image);

            let save_path = format!("data/chars/{}/{}.png", CHARS[i], j);
            cropped.save_with_format(save_path, ImageFormat::Png).unwrap();
        }
    }
}

fn crop_to_fit(image: &image::RgbImage) -> image::RgbImage {
    let mut minx = 0;
    'outer: for x in 0..image.width() {
        for y in 0..image.height() {
            let color = image.get_pixel(x, y);
            if color.0 != [255, 255, 255] {
                minx = x.saturating_sub(1);
                break 'outer;
            }
        }
    }
    let mut miny = 0;
    'outer: for y in 0..image.height() {
        for x in 0..image.width() {
            let color = image.get_pixel(x, y);
            if color.0 != [255, 255, 255] {
                miny = y.saturating_sub(1);
                break 'outer;
            }
        }
    }
    let mut maxx = image.width() - 1;
    {
        let mut x = image.width() - 1;
        'outer: while x > 0 {
            for y in 0..image.height() {
                let color = image.get_pixel(x, y);
                if color.0 != [255, 255, 255] {
                    maxx = maxx.min(x + 1);
                    break 'outer;
                }
            }
            x -= 1;
        }
    }
    let mut maxy = image.height() - 1;
    {
        let mut y = image.height() - 1;
        'outer: while y > 0 {
            for x in 0..image.width() {
                let color = image.get_pixel(x, y);
                if color.0 != [255, 255, 255] {
                    maxy = maxy.min(y + 1);
                    break 'outer;
                }
            }
            y -= 1;
        }
    }
    let cropped = imageops::crop_imm(image, minx, miny, maxx - minx, maxy - miny).to_image();
    cropped
}