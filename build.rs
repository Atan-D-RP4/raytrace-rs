use std::env;
use std::fs;
use std::path::Path;

const MAX_DIMS: usize = 21200;

fn main() {
    println!("cargo::rerun-if-changed=new-joe-kuo-6.21201");
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("sobol_dirs.rs");

    // Read the data file relative to the crate root.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let data_path = Path::new(&manifest_dir).join("new-joe-kuo-6.21201");
    let file = fs::read_to_string(&data_path).expect("Failed to read new-joe-kuo-6.21201");

    let mut directions = [[0u32; 32]; MAX_DIMS];

    // Van der Corput (dim 0)
    for j in 0..32 {
        directions[0][j] = 1u32 << (31 - j);
    }

    for line in file.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 4 {
            continue;
        }
        let d_val: usize = tokens[0].parse().unwrap_or(0);
        let s: usize = tokens[1].parse().unwrap_or(0);
        let a: u32 = tokens[2].parse().unwrap_or(0);
        if !(1..=32).contains(&s) {
            continue;
        }

        let mut m = [0u32; 32];
        for i in 0..s {
            if i + 3 < tokens.len() {
                m[i] = tokens[i + 3].parse().unwrap_or(0);
            }
        }

        let mut v = [0u32; 32];
        for k in 0..s {
            v[k] = m[k] << (32 - (k + 1));
        }
        for k in s..32 {
            let mut val = v[k - s] ^ (v[k - s] >> s);
            for i in 1..s {
                if ((a >> (s - i - 1)) & 1) != 0 {
                    val ^= v[k - i];
                }
            }
            v[k] = val;
        }

        if d_val >= 2 {
            let sob_dim = d_val - 1;
            if sob_dim < MAX_DIMS {
                directions[sob_dim] = v;
            }
        }
    }

    // Write as a Rust const array literal.
    let mut out = String::with_capacity(256 * 1024);
    out.push_str("/// Joe & Kuo 2008 Sobol' direction numbers, computed at compile time.\n");
    out.push_str("pub(super) static DIRS: [[u32; 32]; 21200] = [\n");
    for row in &directions {
        out.push_str("    [");
        for (i, &val) in row.iter().enumerate() {
            out.push_str(&format!("0x{:08X}u32", val));
            if i < 31 {
                out.push_str(", ");
            }
        }
        out.push_str("],\n");
    }
    out.push_str("];\n");

    fs::write(&dest_path, out).expect("Failed to write dirs.rs");
}
