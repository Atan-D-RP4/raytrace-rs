use rayon::prelude::*;
use tracing::info;

use crate::camera::{Camera, CameraSampler};
use crate::film::{Film, FilmTile, SharedFramebuffer};
use crate::hittable::{Intersectable, Sampleable};
use crate::integrator::Integrator;
use crate::renderer::Renderer;
use crate::sampler::{DimCursor, Sampler};
use crate::vec3::Color3;

pub struct CpuRenderer {
    samples_per_pixel: u32,
}

impl CpuRenderer {
    pub fn new(samples_per_pixel: u32) -> Self {
        Self { samples_per_pixel }
    }

    /// Initialize the rayon global thread pool. Must be called before any
    /// `rayon::current_num_threads()` query, otherwise the default pool
    /// (all cores) is created and this call becomes a no-op.
    pub fn init_thread_pool(num_threads: usize) {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global();
    }
}

impl<S, W, L, I, C, F> Renderer<S, W, L, I, C, F> for CpuRenderer
where
    S: Sampler,
    W: Intersectable,
    L: Sampleable<S>,
    I: Integrator<S>,
    C: Camera,
    F: Film,
{
    fn render(
        &self,
        camera: &C,
        integrator: &I,
        film: &mut F,
        world: &W,
        lights: &L,
        framebuffer: Option<SharedFramebuffer>,
        make_sampler: impl Fn(i32, i32) -> S + Sync,
    ) {
        let (width, height) = camera.image_resolution();

        let _span = tracing::info_span!(
            "render",
            width = width,
            height = height,
            spp = self.samples_per_pixel,
            progressive = framebuffer.is_some(),
        )
        .entered();

        info!("camera render started");
        profiling::scope!("camera_render_loop");

        let render_start = std::time::Instant::now();

        let tile_size = 64u32; // Define a tile size for rendering

        // Determine the number of tiles in x and y directions
        let tiles_x = width.div_ceil(tile_size);
        let tiles_y = height.div_ceil(tile_size);

        // Pre-allocate all tiles once, reuse across passes to avoid constant
        // heap allocation churn.
        let mut tile_pool: Vec<FilmTile> = (0..tiles_y * tiles_x)
            .map(|tile_idx| {
                let tx = tile_idx % tiles_x;
                let ty = tile_idx / tiles_x;
                let x_start = tx * tile_size;
                let y_start = ty * tile_size;
                let x_end = (x_start + tile_size).min(width);
                let y_end = (y_start + tile_size).min(height);
                FilmTile::new([x_start, x_end, y_start, y_end])
            })
            .collect();

        for sample_idx in 0..self.samples_per_pixel {
            let pass_start = std::time::Instant::now();
            let _sample_span =
                tracing::info_span!("sample_pass", sample_index = sample_idx + 1).entered();
            profiling::scope!("sample_pass");

            // Zero-fill all pooled tiles before reuse.
            for tile in &mut tile_pool {
                tile.pixels.fill(Color3::ZERO);
            }

            // Parallel Iterate over tiles — each thread produces its own FilmTile,
            // then we merge sequentially to avoid contention on the film.
            tile_pool = tile_pool
                .into_par_iter()
                .enumerate()
                .map(|(_tile_idx, mut tile)| {
                    // Bounds were set at pool creation time; read them from the tile.
                    let [x_start, x_end, y_start, y_end] = tile.bounds;

                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            // Sample the pixel using the sampler and camera
                            let sampler = make_sampler(x as i32, y as i32);
                            let mut dim_cursor = DimCursor::new(0, sampler);
                            dim_cursor.sample_idx = sample_idx;

                            // Generate a camera sample for the pixel from sampler dimensions
                            // Dims 0-1: AA jitter, dims 2-3: lens (defocus)
                            let camera_sampler = CameraSampler {
                                pixel: (x, y),
                                jitter: (dim_cursor.next_sample(), dim_cursor.next_sample()),
                                lens: (dim_cursor.next_sample(), dim_cursor.next_sample()),
                                time: 0.,
                            };

                            if let Some(mut cam_ray) = camera.generate_ray(&camera_sampler) {
                                let radiance =
                                    integrator.li(&mut cam_ray.ray, world, lights, &mut dim_cursor);
                                let sample = radiance * cam_ray.weight;
                                // Guard against NaN/Inf poisoning the accumulation buffer.
                                if sample.x.is_finite()
                                    && sample.y.is_finite()
                                    && sample.z.is_finite()
                                {
                                    tile.add_sample(x, y, sample);
                                }
                            }
                        }
                    }
                    tile
                })
                .collect();

            // Merge all tiles sequentially — fast, no contention.
            for tile in &tile_pool {
                film.merge_tile(tile);
            }

            // Progressive rendering: adaptive cadence to reduce lock contention.
            // Fast early feedback (every pass), then increasingly sparse.
            let should_publish = if framebuffer.is_some() {
                let pass_num = sample_idx + 1;
                let cadence = if pass_num <= 16 {
                    1
                } else if pass_num <= 64 {
                    4
                } else {
                    8
                };
                pass_num % cadence == 0 || pass_num == self.samples_per_pixel
            } else {
                false
            };
            if should_publish && let Some(ref framebuffer) = framebuffer {
                let rgb = film.progressive(sample_idx as usize + 1);
                if let Ok(mut fb) = framebuffer.write() {
                    fb.width = width;
                    fb.height = height;
                    fb.rgb.iter_mut().zip(rgb).for_each(|(dst, src)| *dst = src);
                    fb.finished = sample_idx + 1 == self.samples_per_pixel;
                }
            }

            // Log pass timing every 8 passes and on the final pass.
            if (sample_idx + 1) % 8 == 0 || sample_idx + 1 == self.samples_per_pixel {
                let elapsed = pass_start.elapsed();
                let avg_sec = render_start.elapsed().as_secs_f64() / (sample_idx + 1) as f64;
                info!(
                    sample = sample_idx + 1,
                    total = self.samples_per_pixel,
                    pass_sec = format!("{:.4}", elapsed.as_secs_f64()),
                    avg_sec = format!("{:.4}", avg_sec),
                    eta_sec = format!(
                        "{:.4}",
                        avg_sec * (self.samples_per_pixel - sample_idx - 1) as f64
                    ),
                    "sample pass complete"
                );
            }
        }

        info!(
            elapsed = format!("{:.4}", render_start.elapsed().as_secs_f64()),
            samples_per_sec = format!(
                "{:.2}",
                self.samples_per_pixel as f64 / render_start.elapsed().as_secs_f64()
            ),
            "camera render finished"
        );
    }
}
