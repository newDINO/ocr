mod hand_writer;
mod math_node;

use std::{collections::{HashMap, BTreeMap}, fs::{self, File}, io::{Read, Write}};
use clap::Parser;
use hand_writer::HandWriter;
use math_node::MathNode;
use tiny_skia::{Color, Pixmap};
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
    gen_probs: BTreeMap<String, f32>,
}

fn main() {
    let args = Args::parse();
    let config = read_file_to_string(args.config);
    let config: Config = toml::from_str(&config).unwrap();

    // ensure the sum of the probabilities is 1.0
    assert!(config.gen_probs.iter().fold(0.0, |acc, (_, p)| acc + p) == 1.0);
    
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
            let gen_mode_prob = rng.gen_range(0.0..1.0);
            let mut prob_sum = 0.0;
            let mut chosen_mode = Option::<&str>::None;
            for (mode, prob) in &config.gen_probs {
                prob_sum += prob;
                if gen_mode_prob < prob_sum {
                    chosen_mode = Some(mode);
                    break;
                }
            }
            let chosen_mode = chosen_mode.unwrap();
            if chosen_mode == "normal" {
                nodes.push(MathNode::random(&mut rng, 0, max_depth))
            } else if chosen_mode == "single_char" {
                nodes.push(MathNode::random_single_char(&mut rng))
            }
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


