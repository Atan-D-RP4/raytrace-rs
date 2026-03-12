use std::fs::OpenOptions;
use std::io::Write;

mod vec3;

fn main() {
    let image_width = 256;
    let image_height = 256;

    // Create or open a file
    let file_path = "output.ppm";
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
        .expect("Unable to create or open file");

    file.write_all(format!("P3\n {image_height} {image_width}\n255\n").as_bytes())
        .expect("Unable to write to file");

    let mut output = String::new();

    (0..image_height).for_each(|j| {
        print!("\rScanlines remaining: {}", image_height - j);
        std::io::stdout().flush().unwrap();
        (0..image_width).for_each(|i| {
            let r = i as f32 / (image_width - 1) as f32;
            let g = j as f32 / (image_height - 1) as f32;
            let b = 0.0;

            let ir = (255.999 * r) as i32;
            let ig = (255.999 * g) as i32;
            let ib = (255.999 * b) as i32;

            output.push_str(format!("{ir} {ig} {ib}\n").as_str());
        });
    });

    println!();
    println!("\rDone.");
    file.write_all(output.as_bytes())
        .expect("Unable to write to file");
}
