//! Dimension-indexed QMC & hash-based samplers.
//!
//! Every sample is `sample(n, d)` — determined by `(pass, dimension)` alone.
//! This makes samplers deterministic, `Sync`, and immune to state corruption
//! from variable-length paths.

use std::cell::Cell;
use std::sync::LazyLock;

/// Pure, Sync source of `[0, 1)` samples indexed by pass `n` and dimension `d`.
///
/// Same `(n, d)` always returns the same value — deterministic across threads.
pub trait Sampler: Send + Sync {
    fn sample(&self, n: u32, d: u32) -> f64;
}

const MAX_DIMS: usize = 512;

/// Joe & Kuo 2008 direction numbers, left-aligned u32, lazy-initialized.
static DIRS: LazyLock<[[u32; 32]; MAX_DIMS]> = LazyLock::new(compute_dirs);

/// Parse Joe & Kuo direction numbers from the bundled dataset.
///
/// File format: `d s a m_1 ... m_s` (dimension, degree, primitive poly, values).
fn compute_dirs() -> [[u32; 32]; MAX_DIMS] {
    let file = include_str!("../new-joe-kuo-6.21201");
    let mut dirs = [[0u32; 32]; MAX_DIMS];

    // Van der Corput (dim 0): V[j] = 1 << (31 - j)
    for d in 0..32 {
        dirs[0][d] = 1u32 << (31 - d);
    }

    let mut dim_idx = 2usize; // file starts at dimension 2
    for line in file.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 4 {
            continue;
        }
        let d_val: usize = tokens[0].parse().unwrap_or(0);
        let s: usize = tokens[1].parse().unwrap_or(0);
        let a: u32 = tokens[2].parse().unwrap_or(0);
        if !(1..=32).contains(&s) {
            continue;
        }

        let mut m = [0u32; 32];
        for i in 0..s {
            if i + 3 < tokens.len() {
                m[i] = tokens[i + 3].parse().unwrap_or(0);
            }
        }

        let mut v = [0u32; 32];
        for k in 0..s {
            v[k] = m[k] << (32 - s);
        }
        for k in s..32 {
            let mut val = v[k - s] ^ (v[k - s] >> s);
            for i in 1..s {
                if ((a >> (s - i - 1)) & 1) != 0 {
                    val ^= v[k - i];
                }
            }
            v[k] = val;
        }

        if d_val >= 2 {
            let sob_dim = d_val - 1;
            if sob_dim < MAX_DIMS {
                dirs[sob_dim] = v;
            }
        }
        dim_idx += 1;
        if dim_idx > MAX_DIMS + 1 {
            break;
        }
    }
    dirs
}

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
fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Per-dimension digital shift from `(seed, d)`.
fn splitmix_shift(seed: u64, d: u32) -> u32 {
    splitmix64(seed.wrapping_add(d as u64).wrapping_mul(0x9E3779B97F4A7C15)) as u32
}

/// Hash `(n, d, seed)` into [0, 1).
fn hash_sample(n: u32, d: u32, seed: u64) -> f64 {
    let h = splitmix64(
        seed ^ (n as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(d as u64),
    );
    ((h >> 11) as f64) * INV_53
}

// Gray-code Sobol recurrence: x_{k+1}[d] = x_k[d] ⊕ V[d][tzcnt(k+1)].
// Since V[d][c] only depends on c = tzcnt(k), ALL dimensions share step c.
//
// xs[d] stores raw_accum ⊕ shift(seed, d) so the hot path is cache lookup +
// multiply — no per-sample splitmix.  When the pixel seed changes the cache
// rebases all 512 dims (amortised across 50+ bounces × ~8 dims).

thread_local! {
    static GRAY_CACHE: GrayCodeCache = const { GrayCodeCache {
        last_n: Cell::new(0),
        last_seed: Cell::new(0),
        // Cell<u32> is !Copy, const { ... } initialises each element independently.
        xs: [const { Cell::new(0) }; MAX_DIMS],
    } };
}

/// Per-thread Sobol cache — folds the digital shift into xs[d] so sample()
/// is just a load + multiply.  Rebases all 512 dims when the pixel seed changes.
struct GrayCodeCache {
    last_n: Cell<u32>,
    last_seed: Cell<u64>,
    xs: [Cell<u32>; MAX_DIMS],
}

impl GrayCodeCache {
    /// Ensure cache is current for `(n, seed)`, return `xs[d]`.
    #[inline(always)]
    fn get(&self, d: usize, n: u32, seed: u64) -> u32 {
        let last_seed = self.last_seed.get();
        if last_seed != seed {
            // Rebase: XOR out old shift, XOR in new shift for all 512 dims.
            for dim in 0..MAX_DIMS {
                let cur = self.xs[dim].get();
                let old_sh = splitmix_shift(last_seed, dim as u32);
                let new_sh = splitmix_shift(seed, dim as u32);
                self.xs[dim].set(cur ^ old_sh ^ new_sh);
            }
            self.last_seed.set(seed);
        }

        let last_n = self.last_n.get();
        if n > last_n {
            for k in (last_n + 1)..=n {
                let c = k.trailing_zeros() as usize;
                for dim in 0..MAX_DIMS {
                    let cur = self.xs[dim].get();
                    self.xs[dim].set(cur ^ DIRS[dim][c]);
                }
            }
            self.last_n.set(n);
        } else if n < last_n {
            // Full reset: initialise with shift(seed), then accumulate to n.
            for dim in 0..MAX_DIMS {
                self.xs[dim].set(splitmix_shift(seed, dim as u32));
            }
            for k in 1..=n {
                let c = k.trailing_zeros() as usize;
                for dim in 0..MAX_DIMS {
                    let cur = self.xs[dim].get();
                    self.xs[dim].set(cur ^ DIRS[dim][c]);
                }
            }
            self.last_n.set(n);
        }

        self.xs[d].get()
    }
}

/// Sobol' quasi-random sampler — uses a thread-local Gray-code cache
/// so `sample()` is a single cache lookup + multiply.
pub struct SobolQmcSampler {
    seed: u64,
}

impl SobolQmcSampler {
    pub fn new() -> Self {
        use rand::RngExt;
        Self {
            seed: rand::rng().random(),
        }
    }

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
        Self::new()
    }
}

impl Sampler for SobolQmcSampler {
    #[inline(always)]
    fn sample(&self, n: u32, d: u32) -> f64 {
        GRAY_CACHE.with(|cache| {
            let d_idx = (d as usize).min(MAX_DIMS - 1);
            // xs[d] has the digital shift folded in — no per-sample hash.
            cache.get(d_idx, n, self.seed) as f64 * INV_U32
        })
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

/// Auto-advancing dimension cursor for sequential sample access.
///
/// Replaces `&mut u32` with a safe wrapper that advances on each access,
/// preventing dimension aliasing when callers forget to increment.
///
/// Use with `Sampler::sample(n, cursor.next())` to consume dimensions sequentially:
///
/// ```ignore
/// let mut dims = DimCursor::new(4);
/// let u = sampler.sample(n, dims.next());
/// let v = sampler.sample(n, dims.next());
/// // dims is now at offset 2; next caller starts at dim 6
/// ```
#[derive(Clone, Debug)]
pub struct DimCursor<S: Sampler> {
    base: u32,
    offset: u32,
    pub sample_idx: u32,
    pub sampler: S,
}

impl<S: Sampler> DimCursor<S> {
    /// Creates a cursor starting at dimension `base`.
    #[inline(always)]
    pub fn new(base: u32, sampler: S) -> Self {
        Self {
            base,
            offset: 0,
            sample_idx: 0,
            sampler,
        }
    }

    /// Returns the current dimension and advances by one.
    #[inline(always)]
    pub fn next_dim(&mut self) -> u32 {
        let v = self.base + self.offset;
        self.offset += 1;
        v
    }

    /// Returns next sample and advances dimension.
    #[inline(always)]
    pub fn next_sample(&mut self) -> f64 {
        let d = self.base + self.offset;
        self.offset += 1;
        self.sampler.sample(self.sample_idx, d)
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
    fn sobol_is_sync_and_send() {
        fn assert_sync<T: Sync>() {}
        fn assert_send<T: Send>() {}
        assert_sync::<SobolQmcSampler>();
        assert_send::<SobolQmcSampler>();
        assert_sync::<NaiveRandomSampler>();
        assert_send::<NaiveRandomSampler>();
        assert_sync::<StratifiedRandomSampler>();
        assert_send::<StratifiedRandomSampler>();
    }
}
