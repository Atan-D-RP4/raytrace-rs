//! Dimension-indexed QMC & hash-based samplers.
//!
//! Every sample is `sample(n, d)` — determined by `(pass, dimension)` alone.
//! This makes samplers deterministic, `Sync`, and immune to state corruption
//! from variable-length paths.

/// Pure, Sync source of `[0, 1)` samples indexed by pass `n` and dimension `d`.
///
/// Same `(n, d)` always returns the same value — deterministic across threads.
pub(crate) trait Sampler: Send + Sync {
    fn sample(&self, n: u32, d: u32) -> f64;
}

/// Stateful stream of correlated 2D sample points.
/// Each call advances by one 2D point, without waste.
pub trait SampleStream {
    /// Returns the next 2D sample point as (u, v) in [0, 1)^2.
    fn next_2d(&mut self) -> (f64, f64);
}

/// Stateful source of independent random numbers in [0, 1).
pub trait SamplerRng: Send + Sync {
    fn next(&mut self) -> f64;
}

const MAX_DIMS: usize = 21200;

include!(concat!(env!("OUT_DIR"), "/sobol_dirs.rs"));

/// Conversion from u32 to [0,1).
const INV_U32: f64 = 1.0 / (1u64 << 32) as f64;

/// Conversion from u64 upper bits to [0,1).
const INV_53: f64 = 1.0 / (1u64 << 53) as f64;

/// Deterministic per-pixel seed used by the camera.
/// Same hash as `SobolQmcSampler::for_pixel`.
#[inline]
pub fn pixel_seed(pixel_x: i32, pixel_y: i32) -> u64 {
    splitmix64(pixel_x as u64 ^ 0x9E3779B97F4A7C15)
        .wrapping_mul(0xBF58476D1CE4E5B9)
        .wrapping_add(pixel_y as u64 ^ 0xE5B9A97F4A7C15F0)
}

/// SplitMix64 — fast deterministic hash.
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Per-dimension digital shift from `(seed, d)`.
fn splitmix_shift(seed: u64, d: u32) -> u32 {
    splitmix64(seed.wrapping_add(d as u64).wrapping_mul(0x9E3779B97F4A7C15)) as u32
}

/// Hash `(n, d, seed)` into [0, 1).
pub(crate) fn hash_sample(n: u32, d: u32, seed: u64) -> f64 {
    let h = splitmix64(
        seed ^ (n as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(d as u64),
    );
    ((h >> 11) as f64) * INV_53
}

/// Sobol' quasi-random sampler — stateless, uses direct Gray-code
/// computation so `sample(n, d)` is fully deterministic and independent
/// of prior sample history.
pub struct SobolQmcSampler {
    seed: u64,
}

impl SobolQmcSampler {
    /// Create a sampler with an explicit seed for deterministic results.
    /// For per-pixel deterministic seeding, use [`for_pixel()`](Self::for_pixel).
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }

    /// Deterministic seed from pixel coordinates.
    pub fn for_pixel(pixel_x: i32, pixel_y: i32) -> Self {
        let seed = splitmix64(pixel_x as u64 ^ 0x9E3779B97F4A7C15)
            .wrapping_mul(0xBF58476D1CE4E5B9)
            .wrapping_add(pixel_y as u64 ^ 0xE5B9A97F4A7C15F0);
        Self { seed }
    }
}

impl Default for SobolQmcSampler {
    fn default() -> Self {
        Self::with_seed(0)
    }
}

impl Sampler for SobolQmcSampler {
    #[inline(always)]
    fn sample(&self, n: u32, d: u32) -> f64 {
        if d < MAX_DIMS as u32 {
            let d_idx = d as usize;
            // Gray code g(n) = n ^ (n >> 1).
            // Each set bit at position c contributes V[dim][c].
            let gn = n ^ (n >> 1);
            let mut v = splitmix_shift(self.seed, d);
            let mut g = gn;
            while g != 0 {
                let c = g.trailing_zeros() as usize;
                v ^= DIRS[d_idx][c];
                g &= g - 1; // clear lowest set bit
            }
            v as f64 * INV_U32
        } else {
            // Hash-based fallback for dims >= MAX_DIMS to avoid structured
            // correlation from clamping all overflow dims to the last direction numbers.
            splitmix_shift(self.seed.wrapping_add(n as u64), d) as f64 * INV_U32
        }
    }
}

/// Hash-based random sampler — SplitMix of (n, d, seed).
pub struct NaiveRandomSampler {
    seed: u64,
}

impl NaiveRandomSampler {
    pub fn new() -> Self {
        use rand::RngExt;
        Self {
            seed: rand::rng().random(),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for NaiveRandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for NaiveRandomSampler {
    #[inline(always)]
    fn sample(&self, n: u32, d: u32) -> f64 {
        hash_sample(n, d, self.seed)
    }
}

impl SampleStream for NaiveRandomSampler {
    #[inline(always)]
    fn next_2d(&mut self) -> (f64, f64) {
        let u = self.next();
        let v = self.next();
        (u, v)
    }
}

impl SamplerRng for NaiveRandomSampler {
    #[inline(always)]
    fn next(&mut self) -> f64 {
        let n = self.seed as u32;
        let d = (self.seed >> 32) as u32;
        let v = hash_sample(n, d, self.seed);
        self.seed = self.seed.wrapping_add(1);
        v
    }
}

/// Stratified (jittered) grid for dims 0-1, SplitMix fallback for rest.
pub struct StratifiedRandomSampler {
    seed: u64,
    sqrt_spp: u32,
}

impl StratifiedRandomSampler {
    pub fn new(sqrt_spp: u32, seed: u64) -> Self {
        Self {
            seed,
            sqrt_spp: sqrt_spp.max(1),
        }
    }
}

impl Sampler for StratifiedRandomSampler {
    #[inline(always)]
    fn sample(&self, n: u32, d: u32) -> f64 {
        if d < 2 {
            let cell = n % (self.sqrt_spp * self.sqrt_spp);
            let si = cell / self.sqrt_spp;
            let sj = cell % self.sqrt_spp;
            let jitter = hash_sample(n, d, self.seed);
            let cell_offset = if d == 0 { si } else { sj };
            ((cell_offset as f64) + jitter) * (1.0 / self.sqrt_spp as f64)
        } else {
            hash_sample(n, d, self.seed)
        }
    }
}

impl SampleStream for StratifiedRandomSampler {
    #[inline(always)]
    fn next_2d(&mut self) -> (f64, f64) {
        let n = self.seed as u32;
        self.seed = self.seed.wrapping_add(1);
        (self.sample(n, 0), self.sample(n, 1))
    }
}

impl SamplerRng for StratifiedRandomSampler {
    #[inline(always)]
    fn next(&mut self) -> f64 {
        let n = self.seed as u32;
        self.seed = self.seed.wrapping_add(1);
        self.sample(n, 0)
    }
}

/// Factory for creating per-pixel `SampleStream` instances.
pub trait SampleStreamFactory: Send + Sync {
    type SampleStream: crate::sampler::SampleStream;
    fn for_pixel(&self, x: i32, y: i32, sample_idx: u32) -> Self::SampleStream;
}

/// Factory for creating per-pixel `SamplerRng` instances.
pub trait RngFactory: Send + Sync {
    type Rng: crate::sampler::SamplerRng;
    fn for_pixel(&self, x: i32, y: i32, sample_idx: u32) -> Self::Rng;
}

/// Factory for `SampleStreamWriter` — per-pixel Sobol stream.
pub struct SobolStreamFactory;

impl SampleStreamFactory for SobolStreamFactory {
    type SampleStream = SampleStreamWriter;
    fn for_pixel(&self, x: i32, y: i32, sample_idx: u32) -> SampleStreamWriter {
        SampleStreamWriter::for_pixel(x, y, sample_idx)
    }
}

/// Factory for `HashRng` — per-pixel hash RNG.
pub struct HashRngFactory;

impl RngFactory for HashRngFactory {
    type Rng = HashRng;
    fn for_pixel(&self, x: i32, y: i32, sample_idx: u32) -> HashRng {
        HashRng::for_pixel(x, y, sample_idx)
    }
}

/// Stateful Sobol stream — wraps a stateless `SobolQmcSampler` and advances
/// through 2D dimension pairs. Each `next_2d()` call returns the next
/// correlated (u, v) pair from sequential Sobol dimensions.
pub struct SampleStreamWriter {
    sampler: SobolQmcSampler,
    sample_idx: u32,
    next_pair: u32,
}

impl SampleStreamWriter {
    pub fn new(sampler: SobolQmcSampler, sample_idx: u32) -> Self {
        Self {
            sampler,
            sample_idx,
            next_pair: 0,
        }
    }

    pub fn for_pixel(pixel_x: i32, pixel_y: i32, sample_idx: u32) -> Self {
        Self::new(SobolQmcSampler::for_pixel(pixel_x, pixel_y), sample_idx)
    }
}

impl SampleStream for SampleStreamWriter {
    #[inline(always)]
    fn next_2d(&mut self) -> (f64, f64) {
        let d = self.next_pair * 2;
        let u = self.sampler.sample(self.sample_idx, d);
        let v = self.sampler.sample(self.sample_idx, d + 1);
        self.next_pair += 1;
        (u, v)
    }
}

/// Hash-based independent random number generator.
/// Each call produces an independent value via SplitMix64.
pub struct HashRng {
    seed: u64,
    counter: u32,
}

impl HashRng {
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    pub fn for_pixel(pixel_x: i32, pixel_y: i32, sample_idx: u32) -> Self {
        let seed =
            pixel_seed(pixel_x, pixel_y).wrapping_add(sample_idx as u64 * 0x9E3779B97F4A7C15);
        Self { seed, counter: 0 }
    }
}

impl SamplerRng for HashRng {
    #[inline(always)]
    fn next(&mut self) -> f64 {
        let v = hash_sample(self.counter, 0, self.seed);
        self.counter += 1;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_numbers_are_valid() {
        let dirs = &DIRS;
        assert_eq!(dirs[0][0], 0x80000000);
        assert_eq!(dirs[0][1], 0x40000000);
        assert_eq!(dirs[0][2], 0x20000000);
        assert_ne!(
            dirs[1][0], 0,
            "Dim 1 should have non-zero direction numbers"
        );
        assert_ne!(
            dirs[1][1], 0,
            "Dim 1 should have non-zero direction numbers"
        );
    }

    #[test]
    fn sobol_produces_values_in_unit_interval() {
        let s = SobolQmcSampler::with_seed(42);
        for n in 0..256u32 {
            for d in 0..8u32 {
                let v = s.sample(n, d);
                assert!(
                    (0.0..1.0).contains(&v),
                    "Sobol sample out of range: n={n}, d={d}, v={v}"
                );
            }
        }
    }

    #[test]
    fn sobol_van_der_corput_first_few() {
        let s = SobolQmcSampler { seed: 0 };
        let tol = 1.0 / (1u64 << 32) as f64;
        assert!((s.sample(0, 0) - 0.0).abs() < tol);
        assert!((s.sample(1, 0) - 0.5).abs() < tol);
        assert!((s.sample(2, 0) - 0.75).abs() < tol);
        assert!((s.sample(3, 0) - 0.25).abs() < tol);
        assert!((s.sample(4, 0) - 0.375).abs() < tol);
        assert!((s.sample(5, 0) - 0.875).abs() < tol);
        assert!((s.sample(6, 0) - 0.625).abs() < tol);
        assert!((s.sample(7, 0) - 0.125).abs() < tol);
    }

    #[test]
    fn naive_produces_values_in_unit_interval() {
        let s = NaiveRandomSampler::with_seed(42);
        for n in 0..1000 {
            for d in 0..4 {
                let v = s.sample(n, d);
                assert!((0.0..1.0).contains(&v));
            }
        }
    }

    #[test]
    fn stratified_produces_values_in_unit_interval() {
        let s = StratifiedRandomSampler::new(4, 42);
        for n in 0..256 {
            for d in 0..4 {
                let v = s.sample(n, d);
                assert!((0.0..1.0).contains(&v));
            }
        }
    }

    #[test]
    fn stratified_covers_all_cells_d0() {
        let s = StratifiedRandomSampler::new(4, 42);
        let mut cells = [[false; 4]; 4];
        for n in 0..16 {
            let x = s.sample(n, 0);
            let y = s.sample(n, 1);
            let ci = (x * 4.0) as usize;
            let cj = (y * 4.0) as usize;
            cells[ci][cj] = true;
        }
        assert!(cells.iter().all(|row| row.iter().all(|&c| c)));
    }

    #[test]
    fn deterministic_results() {
        let s1 = SobolQmcSampler::with_seed(12345);
        let s2 = SobolQmcSampler::with_seed(12345);
        for n in 0..64 {
            for d in 0..16 {
                assert_eq!(s1.sample(n, d), s2.sample(n, d));
            }
        }
    }

    #[test]
    fn different_seeds_different_results() {
        let s1 = SobolQmcSampler::with_seed(12345);
        let s2 = SobolQmcSampler::with_seed(54321);
        let mut same = 0;
        for n in 0..64 {
            for d in 0..8 {
                if s1.sample(n, d) == s2.sample(n, d) {
                    same += 1;
                }
            }
        }
        assert!(
            same < 16,
            "Different seeds should produce different samples"
        );
    }

    #[test]
    fn naive_different_pixels_different_samples() {
        let s1 = NaiveRandomSampler::with_seed(pixel_seed(0, 0));
        let s2 = NaiveRandomSampler::with_seed(pixel_seed(10, 10));
        let mut same = 0u32;
        for n in 0..64 {
            for d in 0..8 {
                if (s1.sample(n, d) - s2.sample(n, d)).abs() < f64::EPSILON {
                    same += 1;
                }
            }
        }
        assert!(
            same < 4,
            "Different pixels should produce different samples"
        );
    }

    #[test]
    fn stratified_different_pixels_different_samples() {
        let s1 = StratifiedRandomSampler::new(4, pixel_seed(0, 0));
        let s2 = StratifiedRandomSampler::new(4, pixel_seed(10, 10));
        let mut same = 0u32;
        for n in 0..64 {
            for d in 0..8 {
                if (s1.sample(n, d) - s2.sample(n, d)).abs() < f64::EPSILON {
                    same += 1;
                }
            }
        }
        assert!(
            same < 4,
            "Different pixels should produce different samples"
        );
    }
}
