mod aabb;
mod bvh;
mod camera;
mod hittable;
mod interval;
mod material;
mod perlin;
mod quad;
mod ray;
mod scene;
mod sphere;
mod texture;
mod vec3;

use bvh::BvhNode;
use camera::Camera;
use image::{ImageBuffer, Rgb, RgbImage};
use scene::Scene;

fn main() {
    let scene = Scene::cornell_box();
    let filename = "cornell_box.png";

    let config = *scene.config();
    let mut objects = scene.into_objects();

    let world_len = objects.len();
    let world = std::sync::Arc::new(BvhNode::new(&mut objects, 0, world_len));

    let mut camera = Camera::from_config(&config);

    let start = std::time::Instant::now();
    let (width, height, rgb_data) = camera.render(world);
    let end = std::time::Instant::now();
    println!("Time to render scene: {:?}", end - start);

    let mut img: RgbImage = ImageBuffer::new(width, height);
    for (i, pixel) in rgb_data.chunks(3).enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(x, y, Rgb([pixel[0], pixel[1], pixel[2]]));
    }

    img.save(filename).expect("Failed to save image");

    match std::process::Command::new("satty")
        .args(["--filename", filename])
        .spawn()
    {
        Ok(mut child) => child.wait().map_or_else(
            |e| panic!("Failed to close satty with error: {e:?}"),
            |_| (),
        ),
        Err(e) => panic!("Failed to spawn `satty` with error: {e:?}"),
    }
}
