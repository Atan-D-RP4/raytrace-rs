use rayon::prelude::*;

use crate::camera::Camera;
use crate::camera::get_camera_sample;
use crate::film::{Film, FilmTile, SharedFramebuffer, rgb::FILTER_RADIUS};
use crate::integrator::Integrator;
use crate::intersect::Intersectable;
use crate::math::vec3::Color3;
use crate::primitives::LightPrimitive;
use crate::sampler::{
    HashRng, SampleStreamWriter, morton_encode, owen_scramble_base_4, pixel_sample_state,
};

pub mod cpu;
pub mod wavefront;

pub use cpu::CpuRenderer;
pub use wavefront::WavefrontRenderer;

/// Capability contract for renderers that execute the shared tiled sample loop.
/// Implementations provide the tile execution model; the default methods own
/// tile preparation, deterministic sample indexing, convergence, merging, and
/// progressive publication.
pub(crate) trait TiledRenderer: Send + Sync {
    type Integrator: Integrator;

    fn integrator(&self) -> &Self::Integrator;
    fn samples_per_pixel(&self) -> u32;
    fn threshold_abs(&self) -> f32;
    fn threshold_rel(&self) -> f32;
    fn min_samples_before_adapt(&self) -> u32;
    fn tile_size(&self) -> u32;

    fn render_tile<C, W>(
        &self,
        camera: &C,
        world: &W,
        lights: &[LightPrimitive],
        converged: &[bool],
        pixel_bases: &[u32],
        tile: &mut FilmTile,
        sample_idx: u32,
    ) where
        C: Camera,
        W: Intersectable;

    fn tile_pool(&self, width: u32, height: u32) -> Vec<FilmTile> {
        let tile_size = self.tile_size();
        assert!(tile_size > 0, "tile size must be greater than zero");
        let tiles_x = width.div_ceil(tile_size);
        let tiles_y = height.div_ceil(tile_size);

        (0..tiles_y * tiles_x)
            .map(|tile_idx| {
                let tx = tile_idx % tiles_x;
                let ty = tile_idx / tiles_x;
                let x_start = tx * tile_size;
                let y_start = ty * tile_size;
                let x_end = (x_start + tile_size).min(width);
                let y_end = (y_start + tile_size).min(height);
                FilmTile::new(
                    [
                        x_start.saturating_sub(FILTER_RADIUS),
                        (x_end + FILTER_RADIUS).min(width),
                        y_start.saturating_sub(FILTER_RADIUS),
                        (y_end + FILTER_RADIUS).min(height),
                    ],
                    [x_start, x_end, y_start, y_end],
                )
            })
            .collect()
    }

    fn prepare_tile(&self, tile: &mut FilmTile, converged: &[bool], width: u32) -> bool {
        let [px_start, px_end, py_start, py_end] = tile.pixel_bounds;
        let width = width as usize;
        tile.dirty = (py_start..py_end)
            .any(|y| (px_start..px_end).any(|x| !converged[y as usize * width + x as usize]));
        if tile.dirty {
            tile.pixels.fill(Color3::ZERO);
            tile.sample_count.fill(0);
        }
        tile.dirty
    }

    fn pixel_bases(&self, width: u32, height: u32) -> Vec<u32> {
        let mut bases = Vec::with_capacity((width * height) as usize);
        for py in 0..height {
            for px in 0..width {
                // The fixed base-4 Owen permutation assigns spatially adjacent
                // pixels distant Sobol blocks.
                let scrambled_pixel = owen_scramble_base_4(morton_encode(px, py), 0x12345678);
                bases.push((scrambled_pixel as u64 * self.samples_per_pixel() as u64) as u32);
            }
        }
        bases
    }

    fn render_scalar_tile<C, W>(
        &self,
        camera: &C,
        world: &W,
        lights: &[LightPrimitive],
        converged: &[bool],
        pixel_bases: &[u32],
        tile: &mut FilmTile,
        sample_idx: u32,
    ) where
        C: Camera,
        W: Intersectable,
    {
        let [x_start, x_end, y_start, y_end] = tile.bounds;
        let (width, _) = camera.image_resolution();
        let width_usize = width as usize;

        if !self.prepare_tile(tile, converged, width) {
            return;
        }

        for (y, x) in (y_start..y_end).flat_map(|y| (x_start..x_end).map(move |x| (y, x))) {
            let pixel_idx = y as usize * width_usize + x as usize;
            if converged[pixel_idx] {
                continue;
            }

            let (mut stream, mut rng): (SampleStreamWriter, HashRng) =
                pixel_sample_state(pixel_bases, pixel_idx, sample_idx, x as i32, y as i32);
            let camera_sampler = get_camera_sample((x, y), &mut stream, &mut rng);
            let cam_ray = camera
                .generate_ray_differential(&camera_sampler)
                .or_else(|| camera.generate_ray(&camera_sampler));

            if let Some(mut cam_ray) = cam_ray {
                let radiance =
                    self.integrator()
                        .li(&mut cam_ray.ray, world, lights, &mut stream, &mut rng);
                let sample = radiance * cam_ray.weight;
                if sample.into_inner().is_finite() {
                    tile.add_sample(x, y, sample);
                }
            }
        }
    }

    fn render_tiled<C, F, W>(
        &self,
        camera: &C,
        film: &mut F,
        world: &W,
        lights: &[LightPrimitive],
        framebuffer: Option<SharedFramebuffer>,
    ) where
        C: Camera,
        F: Film,
        W: Intersectable,
        Self: Sized,
    {
        let (width, height) = camera.image_resolution();
        let samples_per_pixel = self.samples_per_pixel();
        let threshold_abs = self.threshold_abs();
        let threshold_rel = self.threshold_rel();
        let min_samples_before_adapt = self.min_samples_before_adapt();

        let _span = tracing::info_span!(
            "render",
            width = width,
            height = height,
            spp = samples_per_pixel,
            progressive = framebuffer.is_some(),
        )
        .entered();

        tracing::info!("camera render started");
        profiling::scope!("camera_render_loop");

        let render_start = std::time::Instant::now();
        let mut tile_pool = self.tile_pool(width, height);
        let mut converged = vec![false; (width * height) as usize];
        let mut pass_times = [0.0f32; 8];
        let mut pass_count: usize = 0;
        let pixel_bases = self.pixel_bases(width, height);

        for sample_idx in 0..samples_per_pixel {
            let pass_start = std::time::Instant::now();
            let _sample_span =
                tracing::info_span!("sample_pass", sample_index = sample_idx + 1).entered();
            profiling::scope!("sample_pass");

            if sample_idx >= min_samples_before_adapt {
                let all_done = film.reset_convergence_mask(
                    threshold_rel,
                    threshold_abs,
                    min_samples_before_adapt,
                    &mut converged,
                );
                if all_done {
                    tracing::info!(
                        "all pixels converged at sample {} of {}",
                        sample_idx + 1,
                        samples_per_pixel
                    );
                    if let Some(ref framebuffer) = framebuffer {
                        let rgb = film.progressive();
                        if let Ok(mut fb) = framebuffer.write() {
                            fb.image
                                .iter_mut()
                                .zip(rgb)
                                .for_each(|(dst, src)| *dst = src);
                            fb.finished = true;
                        }
                    }
                    break;
                }
            } else {
                converged.fill(false);
            }

            tile_pool.par_iter_mut().for_each(|tile| {
                self.render_tile(
                    camera,
                    world,
                    lights,
                    &converged,
                    &pixel_bases,
                    tile,
                    sample_idx,
                );
            });
            tile_pool
                .iter()
                .filter(|tile| tile.dirty)
                .for_each(|tile| film.merge_tile(tile));

            if let Some(ref framebuffer) = framebuffer {
                let pass_num = sample_idx + 1;
                let cadence = match pass_num {
                    1..=16 => 1,
                    17..=64 => 4,
                    _ => 8,
                };
                if pass_num % cadence == 0 || pass_num == samples_per_pixel {
                    let rgb = film.progressive();
                    if let Ok(mut fb) = framebuffer.write() {
                        fb.image
                            .iter_mut()
                            .zip(rgb)
                            .for_each(|(dst, src)| *dst = src);
                        fb.finished = pass_num == samples_per_pixel;
                    }
                }
            }

            if (sample_idx + 1) % 8 == 0 || sample_idx + 1 == samples_per_pixel {
                let elapsed = pass_start.elapsed().as_secs_f32();
                let slot = pass_count % 8;
                pass_times[slot] = elapsed;
                pass_count += 1;
                let window = pass_count.min(8);
                let avg_sec: f32 = pass_times[..window].iter().sum::<f32>() / window as f32;
                tracing::info!(
                    sample = sample_idx + 1,
                    total = samples_per_pixel,
                    pass_sec = format!("{elapsed:.4}"),
                    avg_sec = format!("{avg_sec:.4}"),
                    eta_sec = format!(
                        "{:.4}",
                        avg_sec * (samples_per_pixel - sample_idx - 1) as f32
                    ),
                    "sample pass complete"
                );
            }
        }

        tracing::info!(
            elapsed = format!("{:.4}", render_start.elapsed().as_secs_f32()),
            samples_per_sec = format!(
                "{:.2}",
                samples_per_pixel as f32 / render_start.elapsed().as_secs_f32()
            ),
            "camera render finished"
        );
    }
}

/// A trait for rendering a scene with a given camera and film.
///
/// The renderer is responsible for generating pixel data from the scene geometry and materials,
/// using the camera's projection and the film's output format. It can optionally support
/// progressive rendering by publishing intermediate frames to a shared framebuffer.
pub trait Renderer<C, F>: Send + Sync
where
    C: Camera,
    F: Film,
{
    /// Renders the scene and returns (width, height, RGB pixel data).
    ///
    /// When `framebuffer` is `Some`, publishes progressive intermediate frames
    /// to the shared framebuffer during rendering (live preview mode).
    /// When `None`, renders all samples and returns the final image only.
    fn render(
        &self,
        camera: &C,
        film: &mut F,
        scene: (&impl Intersectable, &[LightPrimitive]),
        framebuffer: Option<SharedFramebuffer>,
    );

    /// Resizes the renderer state for new output dimensions.
    fn resize(&mut self, _width: u32, _height: u32) {}

    /// Resets renderer state for a new scene or camera.
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    use glam::Vec3;

    use crate::camera::perspective::{CameraConfig, PerspectiveCamera};
    use crate::film::RgbFilm;
    use crate::integrator::PathTracingIntegrator;
    use crate::material::{
        DiffuseEmitterMaterial, DiffuseReflector, Material, MicrofacetReflector,
    };
    use crate::math::vec3::{Color3, Direction3, Point3};
    use crate::primitives::Primitive;
    use crate::renderer::{CpuRenderer, WavefrontRenderer};
    use crate::shape::{quad, sphere};

    fn test_camera() -> PerspectiveCamera {
        PerspectiveCamera::from_config(&CameraConfig {
            image_width: 4,
            aspect_ratio: 1.0,
            samples_per_pixel: 2,
            max_depth: 4,
            vfov: 45.0,
            look_from: Point3::new(0.0, 0.0, 4.0),
            look_at: Point3::new(0.0, -0.25, 0.0),
            vup: Direction3::new(0.0, 1.0, 0.0),
            defocus_angle: 0.0,
            focus_distance: 4.0,
            background: Color3::ZERO,
            exposure: 1.0,
            tone_map: false,
        })
    }

    fn test_scene() -> (Vec<Primitive>, Vec<LightPrimitive>) {
        let floor: Primitive = quad(
            Point3::new(-3.0, -1.0, -3.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 6.0),
            Material::from(DiffuseReflector::new(Color3::splat(0.7))),
        )
        .into();
        let light_geometry: Primitive = quad(
            Point3::new(-1.0, 2.0, -2.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
            Material::from(DiffuseEmitterMaterial::new(Color3::splat(8.0))),
        )
        .into();
        let light_sample: LightPrimitive = Primitive::from(quad(
            Point3::new(-1.0, 2.0, -2.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
            Material::from(DiffuseEmitterMaterial::new(Color3::splat(8.0))),
        ))
        .into();
        (vec![floor, light_geometry], vec![light_sample])
    }

    #[test]
    fn wavefront_batch_matches_scalar_for_delta_free_scene() {
        let camera = test_camera();
        let (world, lights) = test_scene();

        let mut cpu_film = RgbFilm::new((4, 4), 1.0, false);
        let mut cpu = CpuRenderer::new(2, PathTracingIntegrator::new(4, Color3::ZERO, None));
        cpu.set_tile_size(4);
        cpu.render(&camera, &mut cpu_film, (&world, &lights), None);

        let mut batch_one_film = RgbFilm::new((4, 4), 1.0, false);
        let mut batch_one =
            WavefrontRenderer::<_, 1>::new(2, PathTracingIntegrator::new(4, Color3::ZERO, None));
        batch_one.set_tile_size(4);
        batch_one.render(&camera, &mut batch_one_film, (&world, &lights), None);

        let mut batch_four_film = RgbFilm::new((4, 4), 1.0, false);
        let mut batch_four =
            WavefrontRenderer::<_, 4>::new(2, PathTracingIntegrator::new(4, Color3::ZERO, None));
        batch_four.set_tile_size(4);
        batch_four.render(&camera, &mut batch_four_film, (&world, &lights), None);

        assert_eq!(cpu_film.read_image(), batch_one_film.read_image());
        assert_eq!(cpu_film.read_image(), batch_four_film.read_image());
    }

    #[test]
    fn wavefront_batch_flows_split_delta_children() {
        let camera = test_camera();
        let mixed = Material::from(MicrofacetReflector::dielectric(Color3::ONE, 0.0, 1.5)).mix(
            Material::from(DiffuseReflector::new(Color3::splat(0.7))),
            0.5,
        );
        let object: Primitive = sphere(Point3::ZERO, 1.0, mixed).into();
        let world = vec![object];

        let mut film = RgbFilm::new((4, 4), 1.0, false);
        let renderer = WavefrontRenderer::<_, 4>::new(
            2,
            PathTracingIntegrator::new(4, Color3::splat(0.1), None),
        );
        renderer.render(&camera, &mut film, (&world, &[]), None);

        let image = film.read_image();
        assert!(image.iter().any(|&channel| channel > 0));
    }
}
