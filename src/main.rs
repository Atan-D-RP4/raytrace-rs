mod aabb;
mod bvh;
mod camera;
mod hittable;
mod interval;
mod material;
mod ray;
mod scene;
mod sphere;
mod texture;
mod vec3;

use std::fs::OpenOptions;
use std::io::Write;

use bvh::BvhNode;
use camera::Camera;
use scene::Scene;

fn main() {
    let scene = Scene::earth_sphere();
    let config = *scene.config();
    let mut objects = scene.into_objects();

    let world_len = objects.len();
    let world = std::sync::Arc::new(BvhNode::new(&mut objects, 0, world_len));

    let mut camera = Camera::from_config(&config);

    let start = std::time::Instant::now();
    let rendered_buffer = camera.render(world);
    let end = std::time::Instant::now();
    println!("Time to render scene: {:?}", end - start);

    let file_path = "earth_sphere.ppm";
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
        .expect("Unable to create or open file");

    println!();
    file.write_all(&rendered_buffer)
        .expect("Unable to write to file");
}
