use image::{ImageBuffer, Rgb, RgbImage};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use raytrace_rs::bvh::BvhNode;
use raytrace_rs::camera::Camera;
use raytrace_rs::scene::Scene;

#[profiling::function]
fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    info!(threads = rayon::current_num_threads(), "startup");

    // TODO(gpu): keep this scene-construction boundary mirrored in future GPU pipeline.
    profiling::scope!("scene_build");

    let scene = Scene::cornell_box();
    let filename = "cornell_box.png";

    let mut config = *scene.config();
    config.samples_per_pixel = 200;

    let mut objects = scene.into_objects();

    let world_len = objects.len();
    info!(object_count = world_len, "building root bvh");
    // TODO(gpu): split accel build from upload/flatten so CPU and GPU can profile same phases.
    profiling::scope!("root_bvh_build");
    let world = std::sync::Arc::new(BvhNode::new(&mut objects, 0, world_len));

    let mut camera = Camera::from_config(&config);

    // let max_threads = rayon::max_num_threads();
    // let _ = rayon::ThreadPoolBuilder::new()
    //     .num_threads(max_threads.checked_sub(0).unwrap_or(max_threads))
    //     .build_global();

    info!(
        threads = rayon::current_num_threads(),
        "rayon threads in use"
    );

    let start = std::time::Instant::now();
    profiling::scope!("render_cpu");
    let (width, height, rgb_data) = camera.render(world);
    let end = std::time::Instant::now();
    info!(elapsed = ?(end - start), width, height, "render complete");

    let mut img: RgbImage = ImageBuffer::new(width, height);
    for (i, pixel) in rgb_data.chunks(3).enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(x, y, Rgb([pixel[0], pixel[1], pixel[2]]));
    }

    info!(%filename, "saving image");
    profiling::scope!("image_save");
    if let Err(error) = img.save(filename) {
        error!(%filename, ?error, "failed to save image");
        return;
    }
    info!(%filename, "image saved");

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
