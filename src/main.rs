mod vec3;
mod ray;

use std::fs::OpenOptions;
use std::io::Write;

use vec3::{Point3, Vec3};

fn write_color(buffer: &mut String, color: Vec3) {
    let icolor = color * 255.999;
    buffer.push_str(
        format!(
            "{} {} {}\n",
            icolor.x as i32, icolor.y as i32, icolor.z as i32
        )
        .as_str(),
    );
}

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
            let color = Vec3 {
                x: i as f64 / (image_width - 1) as f64,
                y: j as f64 / (image_height - 1) as f64,
                z: 0.0,
            };

            write_color(&mut output, color);
        });
    });

    println!();
    println!("\rDone.");
    file.write_all(output.as_bytes())
        .expect("Unable to write to file");
}
