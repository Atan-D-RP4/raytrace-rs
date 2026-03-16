mod camera;
mod hittable;
mod interval;
mod ray;
mod sphere;
mod vec3;

use std::fs::OpenOptions;
use std::io::Write;

use hittable::Hittable;
use interval::Interval;
use sphere::Sphere;
use vec3::Point3;

use self::camera::Camera;

fn main() {
    let image_width = 800;
    let aspect_ratio = 16.0 / 9.0;
    let viewport_height = 5.0;
    let focal_length = 5.0;

    let mut camera = Camera::from(aspect_ratio, image_width, viewport_height, focal_length);

    let world: Vec<Box<dyn Hittable>> = vec![
        Box::new(Sphere::new(&Point3::from(-0.6, 0., -1.), 0.5)),
        Box::new(Sphere::new(&Point3::from(0.6, 0., -1.), 0.5)),
    ];

    let rendered_buffer = camera.render(&world);

    // Create or open a file
    let file_path = "output.ppm";
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
        .expect("Unable to create or open file");

    file.write_all(format!("P3\n{image_width} {}\n255\n", camera.image_height).as_bytes())
        .expect("Unable to write to file");

    println!();
    println!("\rDone.");
    file.write_all(rendered_buffer.as_bytes())
        .expect("Unable to write to file");
}
