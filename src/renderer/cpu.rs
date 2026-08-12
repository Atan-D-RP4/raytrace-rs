use rayon::prelude::*;
use tracing::info;

use crate::camera::{Camera, get_camera_sample};
use crate::film::{Film, FilmTile, SharedFramebuffer, rgb::FILTER_RADIUS};
use crate::integrator::Integrator;
use crate::intersect::Intersectable;
use crate::math::vec3::Color3;
use crate::primitives::LightPrimitive;
use crate::renderer::Renderer;
use crate::sampler::{
    HashRng, SampleStreamWriter, morton_encode, owen_scramble_base_4, pixel_sample_state,
};

pub struct CpuRenderer<I>
where
    I: Integrator,
{
    /// Number of samples to take per pixel. Higher values yield better quality but take longer.
    samples_per_pixel: u32,
    /// Absolute variance floor. Pixels with variance below this threshold are
    /// considered converged regardless of their brightness. Prevents wasting
    /// samples on near-black pixels that are genuinely dark.
    threshold_abs: f32,
    /// Relative variance threshold: variance / luminance². Pixels whose relative
    /// noise drops below this ratio are considered converged. Typical values:
    /// 0.01 (stddev = 10% of mean) to 0.05 (stddev = 22%).
    threshold_rel: f32,
    /// Minimum number of samples to take before considering adaptive sampling.
    /// Ensures we have enough data to make a reliable variance estimate.
    min_samples_before_adapt: u32,
    /// The size of each tile in pixels. Tiles are used to divide the image into smaller regions for
    /// parallel rendering. The tile size should be chosen to balance workload and cache efficiency.
    tile_size: u32,
    /// The integrator used to compute radiance along rays.
    integrator: I,
}

impl<I> CpuRenderer<I>
where
    I: Integrator,
{
    pub fn new(samples_per_pixel: u32, integrator: I) -> Self {
        Self {
            samples_per_pixel,
            threshold_abs: 1e-4,
            threshold_rel: 0.02,
            min_samples_before_adapt: 64,
            tile_size: 32,
            integrator,
        }
    }

    pub fn set_threshold_abs(&mut self, threshold: f32) {
        self.threshold_abs = threshold;
    }

    pub fn set_threshold_rel(&mut self, threshold: f32) {
        self.threshold_rel = threshold;
    }

    pub fn set_min_samples_before_adapt(&mut self, min_samples: u32) {
        self.min_samples_before_adapt = min_samples;
    }

    pub fn set_tile_size(&mut self, tile_size: u32) {
        self.tile_size = tile_size;
    }

    /// Initializes a pool of film tiles for rendering. Each tile covers a portion of the image, and
    /// the pool is used to distribute work across threads. The tiles are sized to fit within the
    /// image dimensions, and any remaining pixels are handled by smaller tiles.
    pub fn tile_pool_init(&self, width: u32, height: u32) -> Vec<FilmTile> {
        let tile_size = self.tile_size;

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
}

impl<I> CpuRenderer<I>
where
    I: Integrator,
{
    /// Render a single tile of the image. This function is called in parallel for each tile.
    fn render_tile<C: Camera>(
        &self,
        camera: &C,
        world: &impl Intersectable,
        lights: &[LightPrimitive],
        converged: &[bool],
        pixel_bases: &[u32],
        tile: &mut FilmTile,
        sample_idx: u32,
    ) {
        let [x_start, x_end, y_start, y_end] = tile.bounds;

        let (width, _) = camera.image_resolution();
        let width = width as usize;

        // Does any pixel in this tile still need samples? Uses the tile's original
        // (unexpanded) pixel bounds: the expanded `bounds` overlap neighboring
        // tiles by FILTER_RADIUS and cannot be inverted at image edges (clamping).
        let [px_start, px_end, py_start, py_end] = tile.pixel_bounds;
        tile.dirty = (py_start..py_end)
            .any(|y| (px_start..px_end).any(|x| !converged[y as usize * width + x as usize]));
        if !tile.dirty {
            return; // Fully converged — no samples needed this pass.
        }

        // Zero-fill the staging buffers for reuse (this tile was merged last pass).
        tile.pixels.fill(Color3::ZERO);
        tile.sample_count.fill(0);

        for (y, x) in (y_start..y_end).flat_map(|y| (x_start..x_end).map(move |x| (y, x))) {
            let pixel_idx = y as usize * width + x as usize;
            if converged[pixel_idx] {
                continue; // Skip pixels that have already converged
            }

            // Fresh stream & rng for this pixel-sample. Construction
            // is cheap (Copy value types) — no pooling needed.
            let (mut stream, mut rng): (SampleStreamWriter, HashRng) =
                pixel_sample_state(pixel_bases, pixel_idx, sample_idx, x as i32, y as i32);

            // Generate a camera sample from the stream & rng.
            let camera_sampler = get_camera_sample((x, y), &mut stream, &mut rng);

            let cam_ray = camera
                .generate_ray_differential(&camera_sampler)
                .or_else(|| camera.generate_ray(&camera_sampler));
            if let Some(mut cam_ray) = cam_ray {
                let radiance =
                    self.integrator
                        .li(&mut cam_ray.ray, world, lights, &mut stream, &mut rng);
                let sample = radiance * cam_ray.weight;
                // Guard against NaN/Inf poisoning the accumulation buffer.
                if sample.into_inner().is_finite() {
                    tile.add_sample(x, y, sample);
                }
            }
        }
    }
}

impl<I, C, F> Renderer<C, F> for CpuRenderer<I>
where
    I: Integrator,
    C: Camera,
    F: Film,
{
    fn render(
        &self,
        camera: &C,
        film: &mut F,
        scene: (&impl Intersectable, &[LightPrimitive]),
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

        let mut tile_pool = self.tile_pool_init(width, height);

        // Pre-allocate the convergence mask once, then refill in place each pass
        // to avoid repeated heap allocation of the full-resolution bool buffer.
        let mut converged = vec![false; (width * height) as usize];

        // Ring buffer for rolling average of last 8 pass durations.
        let mut pass_times = [0.0f32; 8];
        let mut pass_count: usize = 0;

        // Precompute each pixel's base Sobol index once — pass-invariant.
        let mut pixel_bases: Vec<u32> = Vec::with_capacity((width * height) as usize);
        for py in 0..height {
            for px in 0..width {
                // The Morton code orders pixels along a space-filling curve; Owen-scrambling it (base-4) is
                // a *fixed* permutation that shuffles which pixel draws from which contiguous block of
                // `spp` Sobol samples, decorrelating spatially adjacent pixels.
                // The constant is baked in: it's a deterministic permutation, not a tunable seed.
                let scrambled_pixel = owen_scramble_base_4(morton_encode(px, py), 0x12345678);
                pixel_bases.push((scrambled_pixel as u64 * self.samples_per_pixel as u64) as u32);
            }
        }

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

            // Render each tile in parallel. render_tile determines tile dirty-ness
            // from the tile's original pixel bounds, zero-fills its staging buffers,
            // and early-returns for fully-converged tiles (no memset, no samples).
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

            // Merge only dirty tiles — skip fully-converged tiles (no new samples).
            tile_pool
                .iter()
                .filter(|tile| tile.dirty)
                .for_each(|tile| film.merge_tile(tile));

            // Progressive rendering: adaptive cadence to reduce lock contention.
            // Fast early feedback (every pass), then increasingly sparse.
            if let Some(ref framebuffer) = framebuffer {
                let should_publish = {
                    let pass_num = sample_idx + 1;
                    let cadence = match pass_num {
                        1..=16 => 1,
                        17..=64 => 4,
                        _ => 8,
                    };
                    pass_num % cadence == 0 || pass_num == self.samples_per_pixel
                };
                if !should_publish {
                    continue;
                }
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
                let elapsed = pass_start.elapsed().as_secs_f32();
                let slot = pass_count % 8;
                pass_times[slot] = elapsed;
                pass_count += 1;
                let window = pass_count.min(8);
                let avg_sec: f32 = pass_times[..window].iter().sum::<f32>() / window as f32;
                info!(
                    sample = sample_idx + 1,
                    total = self.samples_per_pixel,
                    pass_sec = format!("{:.4}", elapsed),
                    avg_sec = format!("{:.4}", avg_sec),
                    eta_sec = format!(
                        "{:.4}",
                        avg_sec * (self.samples_per_pixel - sample_idx - 1) as f32
                    ),
                    "sample pass complete"
                );
            }
        }

        info!(
            elapsed = format!("{:.4}", render_start.elapsed().as_secs_f32()),
            samples_per_sec = format!(
                "{:.2}",
                self.samples_per_pixel as f32 / render_start.elapsed().as_secs_f32()
            ),
            "camera render finished"
        );
    }
}

pub struct WavefrontRenderer<I, const B: usize>
where
    I: Integrator,
{
    /// Number of samples to take per pixel. Higher values yield better quality but take longer.
    samples_per_pixel: u32,
    /// Absolute variance floor. Pixels with variance below this threshold are
    /// considered converged regardless of their brightness. Prevents wasting
    /// samples on near-black pixels that are genuinely dark.
    threshold_abs: f32,
    /// Relative variance threshold: variance / luminance². Pixels whose relative
    /// noise drops below this ratio are considered converged. Typical values:
    /// 0.01 (stddev = 10% of mean) to 0.05 (stddev = 22%).
    threshold_rel: f32,
    /// Minimum number of samples to take before considering adaptive sampling.
    /// Ensures we have enough data to make a reliable variance estimate.
    min_samples_before_adapt: u32,
    /// The integrator used to compute radiance along rays.
    integrator: I,
}

impl<I, const B: usize> WavefrontRenderer<I, B>
where
    I: Integrator,
{
    pub fn new(
        samples_per_pixel: u32,
        threshold_abs: f32,
        threshold_rel: f32,
        min_samples_before_adapt: u32,
        integrator: I,
    ) -> Self {
        Self {
            samples_per_pixel,
            threshold_abs,
            threshold_rel,
            min_samples_before_adapt,
            integrator,
        }
    }
}

impl<I, C, F, const B: usize> Renderer<C, F> for WavefrontRenderer<I, B>
where
    I: Integrator,
    C: Camera,
    F: Film,
{
    fn render(
        &self,
        _camera: &C,
        _film: &mut F,
        _scene: (&impl Intersectable, &[LightPrimitive]),
        _framebuffer: Option<SharedFramebuffer>,
    ) {
    }
}
