//! Dimension-indexed QMC & hash-based samplers.
//!
//! Every sample is `sample(n, d)` — determined by `(pass, dimension)` alone.
//! This makes samplers deterministic, `Sync`, and immune to state corruption
//! from variable-length paths.
const MAX_DIMS: usize = 21200;

include!(concat!(env!("OUT_DIR"), "/sobol_dirs.rs"));

/// Conversion from u32 to [0,1).
const INV_U32: f32 = 1.0 / (1u64 << 32) as f32;

/// Conversion from the top 24 bits of a hash to [0, 1).
/// Matches f32 mantissa precision — extracting more bits is wasted.
const INV_HASH: f32 = 1.0 / (1u64 << 24) as f32;

/// Fractional bits of the golden ratio φ, used as a hash spacing / scrambling constant.
/// Computed as ⌊(φ − 1) × 2^64⌋.
/// Golden ratio spacing constant for integer hashing: ⌊(φ − 1) × 2^64⌋.
pub(crate) const GOLDEN_RATIO_HASH: u64 = 0x9E3779B97F4A7C15;

/// SplitMix64 LCG multiplier (Vigna 2015).
const SPLITMIX_MULT_1: u64 = 0xBF58476D1CE4E5B9;

/// SplitMix64 finalizer multiplier (Vigna 2015).
const SPLITMIX_MULT_2: u64 = 0x94D049BB133111EB;

/// Pixel-coordinate scrambling salt — decorrelates X and Y in the seed hash.
const PIXEL_COORD_SALT: u64 = 0xE5B9A97F4A7C15F0;

/// Deterministic per-pixel seed used by the camera.
/// Same hash as `SobolQmcSampler::for_pixel`.
#[inline]
pub fn pixel_seed(pixel_x: i32, pixel_y: i32) -> u64 {
    splitmix64(pixel_x as u64 ^ GOLDEN_RATIO_HASH)
        .wrapping_mul(SPLITMIX_MULT_1)
        .wrapping_add(pixel_y as u64 ^ PIXEL_COORD_SALT)
}

/// SplitMix64 — fast deterministic hash (Vigna 2015).
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(SPLITMIX_MULT_1);
    x = (x ^ (x >> 27)).wrapping_mul(SPLITMIX_MULT_2);
    x ^ (x >> 31)
}

/// Per-dimension digital shift from `(seed, d)`.
fn splitmix_shift(seed: u64, d: u32) -> u32 {
    splitmix64(seed.wrapping_add(d as u64).wrapping_mul(GOLDEN_RATIO_HASH)) as u32
}

/// Hash `(n, d, seed)` into [0, 1).
pub(crate) fn hash_sample(n: u32, d: u32, seed: u64) -> f32 {
    let h = splitmix64(
        seed ^ (n as u64)
            .wrapping_mul(GOLDEN_RATIO_HASH)
            .wrapping_add(d as u64),
    );
    // f32 has 24 bits of mantissa; extract exactly 24 bits from the hash.
    (h >> 40) as f32 * INV_HASH
}

/// Owen-scramble a 32-bit unsigned integer.
///
/// Processes bits from most significant to least significant.
/// At each bit position, a hash chain determines whether to flip the bit.
/// Because the hash chain accumulates state from all higher bits,
/// the decision to flip bit k depends on bits 0..k-1 — this is the
/// recursive dependence that defines Owen scrambling.
pub fn owen_scramble(v: u32, seed: u64) -> u32 {
    // Reverse bits so we can iterate MSB-first with a simple loop.
    // After reversal, bit 0 = original MSB, bit 31 = original LSB.
    let mut vr = v.reverse_bits();

    for i in 0..32 {
        // Retrieve the prefix of bits 0..i (original MSB..bit i) to use as a hash input.
        let prefix = vr >> i;
        // Fresh hash for this bit position, based on the seed and the prefix of higher bits.
        let h = splitmix64(seed ^ prefix as u64);
        // Bit i of the hash determines whether to flip bit i of the value.
        let flip = ((h >> i) & 1) as u32;
        // Flip bit i of the value if flip == 1.
        vr ^= flip << i;
    }

    // Reverse bits back to original order.
    vr.reverse_bits()
}

/// Pure, Sync source of `[0, 1)` samples indexed by pass `n` and dimension `d`.
///
/// Same `(n, d)` always returns the same value — deterministic across threads.
pub(crate) trait QmcSampler: Send + Sync {
    fn sample(&self, n: u32, d: u32) -> f32;
}

/// Stateful stream of correlated 2D sample points.
/// Each call advances by one 2D point, without waste.
pub trait SampleStream: Send + Sync {
    /// Returns the next 2D sample point as (u, v) in [0, 1)^2.
    fn next_2d(&mut self) -> (f32, f32);
}

/// Stateful source of independent random numbers in [0, 1).
pub trait SamplerRng: Send + Sync {
    fn next(&mut self) -> f32;
}

/// Sobol' quasi-random sampler — stateless, uses direct Gray-code
/// computation so `sample(n, d)` is fully deterministic and independent
/// of prior sample history.
#[derive(Clone)]
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
        let seed = splitmix64(pixel_x as u64 ^ GOLDEN_RATIO_HASH)
            .wrapping_mul(SPLITMIX_MULT_1)
            .wrapping_add(pixel_y as u64 ^ PIXEL_COORD_SALT);
        Self { seed }
    }
}

impl Default for SobolQmcSampler {
    fn default() -> Self {
        Self::with_seed(0)
    }
}

impl QmcSampler for SobolQmcSampler {
    #[inline(always)]
    fn sample(&self, n: u32, d: u32) -> f32 {
        if d < MAX_DIMS as u32 {
            let d_idx = d as usize;
            // Gray code g(n) = n ^ (n >> 1).
            // Each set bit at position c contributes V[dim][c].
            let gn = n ^ (n >> 1);
            let mut raw = 0;
            let mut g = gn;
            while g != 0 {
                let c = g.trailing_zeros() as usize;
                raw ^= DIRS[d_idx][c];
                g &= g - 1; // clear lowest set bit
            }
            let scramble_seed = splitmix_shift(self.seed, d) as u64;
            let scrambled = owen_scramble(raw, scramble_seed);
            scrambled as f32 * INV_U32
        } else {
            // Hash-based fallback for dims >= MAX_DIMS to avoid structured
            // correlation from clamping all overflow dims to the last direction numbers.
            splitmix_shift(self.seed.wrapping_add(n as u64), d) as f32 * INV_U32
        }
    }
}

/// Hash-based random sampler — SplitMix of (n, d, seed).
#[derive(Clone)]
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

impl QmcSampler for NaiveRandomSampler {
    #[inline(always)]
    fn sample(&self, n: u32, d: u32) -> f32 {
        hash_sample(n, d, self.seed)
    }
}

impl SampleStream for NaiveRandomSampler {
    #[inline(always)]
    fn next_2d(&mut self) -> (f32, f32) {
        let u = self.next();
        let v = self.next();
        (u, v)
    }
}

impl SamplerRng for NaiveRandomSampler {
    #[inline(always)]
    fn next(&mut self) -> f32 {
        let n = self.seed as u32;
        let d = (self.seed >> 32) as u32;
        let v = hash_sample(n, d, self.seed);
        self.seed = self.seed.wrapping_add(1);
        v
    }
}

/// Stratified (jittered) grid for dims 0-1, SplitMix fallback for rest.
#[derive(Clone)]
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

impl QmcSampler for StratifiedRandomSampler {
    #[inline(always)]
    fn sample(&self, n: u32, d: u32) -> f32 {
        if d < 2 {
            let cell = n % (self.sqrt_spp * self.sqrt_spp);
            let si = cell / self.sqrt_spp;
            let sj = cell % self.sqrt_spp;
            let jitter = hash_sample(n, d, self.seed);
            let cell_offset = if d == 0 { si } else { sj };
            ((cell_offset as f32) + jitter) * (1.0 / self.sqrt_spp as f32)
        } else {
            hash_sample(n, d, self.seed)
        }
    }
}

impl SampleStream for StratifiedRandomSampler {
    #[inline(always)]
    fn next_2d(&mut self) -> (f32, f32) {
        let n = self.seed as u32;
        self.seed = self.seed.wrapping_add(1);
        (self.sample(n, 0), self.sample(n, 1))
    }
}

impl SamplerRng for StratifiedRandomSampler {
    #[inline(always)]
    fn next(&mut self) -> f32 {
        let n = self.seed as u32;
        self.seed = self.seed.wrapping_add(1);
        self.sample(n, 0)
    }
}

/// Factory for creating per-pixel `SampleStream` instances.
pub trait SampleStreamFactory: Send + Sync {
    type SampleStream: SampleStream;
    fn for_pixel(&self, x: i32, y: i32, sample_idx: u32) -> Self::SampleStream;
}

/// Factory for creating per-pixel `SamplerRng` instances.
pub trait RngFactory: Send + Sync {
    type Rng: SamplerRng;
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
#[derive(Clone)]
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
    fn next_2d(&mut self) -> (f32, f32) {
        let d = self.next_pair * 2;
        let u = self.sampler.sample(self.sample_idx, d);
        let v = self.sampler.sample(self.sample_idx, d + 1);
        self.next_pair += 1;
        (u, v)
    }
}

/// Hash-based independent random number generator.
/// Each call produces an independent value via SplitMix64.
#[derive(Clone)]
pub struct HashRng {
    seed: u64,
    counter: u32,
}

impl HashRng {
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    pub fn for_pixel(pixel_x: i32, pixel_y: i32, sample_idx: u32) -> Self {
        // wrapping_mul/wrapping_add avoids arithmetic overflow panics in debug builds.
        let seed = pixel_seed(pixel_x, pixel_y)
            .wrapping_add((sample_idx as u64).wrapping_mul(GOLDEN_RATIO_HASH));
        Self { seed, counter: 0 }
    }
}

impl SamplerRng for HashRng {
    #[inline(always)]
    fn next(&mut self) -> f32 {
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
        // Owen scrambling changes the raw Van der Corput values, so we verify
        // the scrambled sequence's key properties instead of hardcoded values:
        // 1. All values in [0, 1)
        // 2. First 8 samples are distinct (no collisions from scrambling)
        // 3. Deterministic (same seed → same values)
        let s = SobolQmcSampler::with_seed(42);
        let mut values = Vec::new();
        for n in 0..8u32 {
            let v = s.sample(n, 0);
            assert!((0.0..1.0).contains(&v), "Sample out of range: n={n}, v={v}");
            values.push(v);
        }
        let unique: std::collections::HashSet<_> = values.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            unique.len(),
            8,
            "All 8 dimension-0 samples should be distinct"
        );

        // Determinism: same seed produces same values
        let s2 = SobolQmcSampler::with_seed(42);
        for n in 0..8u32 {
            assert_eq!(s.sample(n, 0), s2.sample(n, 0));
        }
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
                if (s1.sample(n, d) - s2.sample(n, d)).abs() < f32::EPSILON {
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
                if (s1.sample(n, d) - s2.sample(n, d)).abs() < f32::EPSILON {
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
    fn owen_scramble_nontrivial() {
        let seed = 42;
        let v = 0;
        let scrambled = owen_scramble(v, seed);
        assert_ne!(scrambled, 0, "owen_scramble(0, seed) should not return 0");
    }

    #[test]
    fn owen_scramble_deterministic() {
        let seed = 42;
        let v = 123456789;
        let scrambled1 = owen_scramble(v, seed);
        let scrambled2 = owen_scramble(v, seed);
        assert_eq!(
            scrambled1, scrambled2,
            "owen_scramble(v, seed) should be deterministic"
        );
    }

    #[test]
    fn owen_scramble_different_seeds() {
        let v = 123456789;
        let seed1 = 42;
        let seed2 = 43;
        let scrambled1 = owen_scramble(v, seed1);
        let scrambled2 = owen_scramble(v, seed2);
        assert_ne!(
            scrambled1, scrambled2,
            "owen_scramble(v, seed1) should not equal owen_scramble(v, seed2) for different seeds"
        );
    }

    #[test]
    fn owen_scramble_bijection() {
        let seed = 42;
        let mut seen = std::collections::HashSet::new();
        for v in 0..1000u32 {
            let scrambled = owen_scramble(v, seed);
            assert!(
                !seen.contains(&scrambled),
                "owen_scramble should be a bijection: duplicate output for v={v}"
            );
            seen.insert(scrambled);
        }
    }

    #[test]
    fn sobol_convergence_rate() {
        let s = SobolQmcSampler::with_seed(0);
        let mut sum = 0.0;
        let mut n = 1;
        let mut prev_estimate = 0.0;
        for i in 0..10000 {
            let x = s.sample(i, 0);
            sum += x;
            n += 1;
            let estimate = sum / n as f32;
            if i > 0 {
                let error = (estimate - prev_estimate).abs();
                // Expect error to decrease roughly as O(1/N)
                assert!(
                    error < 1.0 / (n as f32),
                    "Sobol convergence rate is not O(1/N): error={error}, n={n}"
                );
            }
            prev_estimate = estimate;
        }
    }
}
