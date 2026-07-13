use std::sync::Arc;

use rayon::prelude::*;
use tracing::info;

use crate::camera::{Camera, get_camera_sample};
use crate::film::{Film, FilmTile, SharedFramebuffer, rgb::FILTER_RADIUS};
use crate::hittable::{Intersectable, Sampleable};
use crate::integrator::Integrator;
use crate::renderer::Renderer;
use crate::sampler::{self, Sampler};
use crate::vec3::Color3;

pub struct CpuRenderer<I, S>
where
    I: Integrator<S>,
    S: Sampler + Sync,
{
    /// Number of samples to take per pixel. Higher values yield better quality but take longer.
    samples_per_pixel: u32,
    /// Absolute variance floor. Pixels with variance below this threshold are
    /// considered converged regardless of their brightness. Prevents wasting
    /// samples on near-black pixels that are genuinely dark.
    threshold_abs: f64,
    /// Relative variance threshold: variance / luminance². Pixels whose relative
    /// noise drops below this ratio are considered converged. Typical values:
    /// 0.01 (stddev = 10% of mean) to 0.05 (stddev = 22%).
    threshold_rel: f64,
    /// Minimum number of samples to take before considering adaptive sampling.
    /// Ensures we have enough data to make a reliable variance estimate.
    min_samples_before_adapt: u32,
    /// The integrator used to compute radiance along rays.
    integrator: I,
    /// Prototype sampler — cloned once per rayon thread via `ThreadLocal`.
    sampler_prototype: S,
}

impl<I, S> CpuRenderer<I, S>
where
    I: Integrator<S>,
    S: Sampler + Sync,
{
    pub fn new(samples_per_pixel: u32, integrator: I, sampler_prototype: S) -> Self {
        Self {
            samples_per_pixel,
            threshold_abs: 1e-4,
            threshold_rel: 0.02,
            min_samples_before_adapt: 64,
            integrator,
            sampler_prototype,
        }
    }

    pub fn set_threshold_abs(&mut self, threshold: f64) {
        self.threshold_abs = threshold;
    }

    pub fn set_threshold_rel(&mut self, threshold: f64) {
        self.threshold_rel = threshold;
    }

    pub fn set_min_samples_before_adapt(&mut self, min_samples: u32) {
        self.min_samples_before_adapt = min_samples;
    }
}

impl<W, I, C, F, S> Renderer<W, C, F> for CpuRenderer<I, S>
where
    W: Intersectable,
    I: Integrator<S>,
    C: Camera,
    F: Film,
    S: Sampler + Sync,
{
    fn render(
        &self,
        camera: &C,
        film: &mut F,
        scene: (&W, &[Arc<dyn Sampleable>]),
        framebuffer: Option<SharedFramebuffer>,
    ) {
        let (width, height) = camera.image_resolution();
        let (world, lights) = scene;

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
                FilmTile::new([
                    x_start.saturating_sub(FILTER_RADIUS),
                    (x_end + FILTER_RADIUS).min(width),
                    y_start.saturating_sub(FILTER_RADIUS),
                    (y_end + FILTER_RADIUS).min(height),
                ])
            })
            .collect();

        // Pre-allocate the convergence mask once, then refill in place each pass
        // to avoid repeated heap allocation of the full-resolution bool buffer.
        let mut converged = vec![false; (width * height) as usize];

        // Ring buffer for rolling average of last 8 pass durations.
        let mut pass_times = [0.0f64; 8];
        let mut pass_count: usize = 0;

        for sample_idx in 0..self.samples_per_pixel {
            let pass_start = std::time::Instant::now();
            let _sample_span =
                tracing::info_span!("sample_pass", sample_index = sample_idx + 1).entered();
            profiling::scope!("sample_pass");

            // After accumulating enough samples (min_samples_before_adapt), refill
            // the convergence mask in place — no reallocation. Before that, keep
            // all pixels unconverged (false) to build up initial variance estimates.
            if sample_idx >= self.min_samples_before_adapt {
                let all_done = film.reset_convergence_mask(
                    self.threshold_rel,
                    self.threshold_abs,
                    self.min_samples_before_adapt,
                    &mut converged,
                );

                // Early exit: if every pixel has converged, stop sampling.
                if all_done {
                    info!(
                        "all pixels converged at sample {} of {}",
                        sample_idx + 1,
                        self.samples_per_pixel
                    );
                    // Still publish the final frame so the preview shows the result.
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

            // Zero-fill pooled tiles before reuse — skip tiles where all pixels are
            // already converged (avoids 15.7MB of useless memsets per pass).
            for tile in &mut tile_pool {
                let [x_start, _, y_start, _] = tile.bounds;
                let (orig_x_start, orig_y_start) = (
                    x_start.saturating_add(FILTER_RADIUS),
                    y_start.saturating_add(FILTER_RADIUS),
                );
                let (orig_x_end, orig_y_end) = (
                    (orig_x_start + tile_size).min(width),
                    (orig_y_start + tile_size).min(height),
                );

                tile.dirty = (orig_y_start..orig_y_end)
                    .zip(orig_x_start..orig_x_end)
                    .any(|(y, x)| !converged[y as usize * width as usize + x as usize]);

                if tile.dirty {
                    tile.pixels.fill(Color3::ZERO);
                    tile.sample_count.fill(0);
                }
            }

            // Per-thread sampler instances — clone-once reuse via thread index.
            // This replaces the old per-pixel `for_pixel()` factory allocation.
            let num_threads = rayon::current_num_threads();
            let samplers: Vec<std::sync::Mutex<S>> = (0..num_threads)
                .map(|_| std::sync::Mutex::new(self.sampler_prototype.clone()))
                .collect();

            // Parallelly Iterate over tiles — each thread produces its own FilmTile,
            // then we merge sequentially to avoid contention on the film.
            tile_pool
                .par_iter_mut()
                .enumerate()
                .for_each(|(_tile_idx, tile)| {
                    // Bounds were set at pool creation time; read them from the tile.
                    let [x_start, x_end, y_start, y_end] = tile.bounds;

                    let thread_idx = rayon::current_thread_index()
                        .expect("tile processing always runs inside rayon thread pool");
                    let mut sampler_guard = samplers[thread_idx].lock().unwrap();

                    for (y, x) in
                        (y_start..y_end).flat_map(|y| (x_start..x_end).map(move |x| (y, x)))
                    {
                        if converged[y as usize * width as usize + x as usize] {
                            continue; // Skip pixels that have already converged
                        }

                        // Start a new pixel-session — reinitialises per-pixel state
                        // (Sobol seed) from `(x, y, sample_idx)`.
                        let mut session = sampler_guard.begin_pixel(
                            sampler::Point2i {
                                x: x as i32,
                                y: y as i32,
                            },
                            sample_idx,
                        );

                        // Generate a camera sample from session (AA jitter, lens, time)
                        let camera_sampler = get_camera_sample((x, y), &mut session);

                        let cam_ray = camera
                            .generate_ray_differential(&camera_sampler)
                            .or_else(|| camera.generate_ray(&camera_sampler));
                        if let Some(mut cam_ray) = cam_ray {
                            let radiance =
                                self.integrator
                                    .li(&mut cam_ray.ray, world, lights, &mut session);
                            let sample = radiance * cam_ray.weight;
                            // Guard against NaN/Inf poisoning the accumulation buffer.
                            if sample.is_finite() {
                                tile.add_sample(x, y, sample);
                            }
                        }
                        // `session` dropped here — releases the per-pixel borrow
                    }
                    // `sampler_guard` dropped here — releases the per-thread lock
                });

            // Merge only dirty tiles — skip fully-converged tiles (no new samples).
            for tile in &tile_pool {
                if tile.dirty {
                    film.merge_tile(tile);
                }
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
                let rgb = film.progressive();
                if let Ok(mut fb) = framebuffer.write() {
                    fb.image
                        .iter_mut()
                        .zip(rgb)
                        .for_each(|(dst, src)| *dst = src);
                    fb.finished = sample_idx + 1 == self.samples_per_pixel;
                }
            }

            // Log pass timing every 8 passes and on the final pass.
            if (sample_idx + 1) % 8 == 0 || sample_idx + 1 == self.samples_per_pixel {
                let elapsed = pass_start.elapsed().as_secs_f64();
                let slot = pass_count % 8;
                pass_times[slot] = elapsed;
                pass_count += 1;
                let window = pass_count.min(8);
                let avg_sec: f64 = pass_times[..window].iter().sum::<f64>() / window as f64;
                info!(
                    sample = sample_idx + 1,
                    total = self.samples_per_pixel,
                    pass_sec = format!("{:.4}", elapsed),
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
