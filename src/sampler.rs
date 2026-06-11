use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Produces [0, 1) samples for Monte Carlo integration.
/// Must be `Send` for rayon parallel iterators.
pub trait Sampler: Send {
    fn get_next_1d(&mut self) -> f64;
    fn get_next_2d(&mut self) -> [f64; 2];

    /// Fork independent samplers. Parent state is preserved.
    fn split(&mut self, num: usize) -> Vec<Box<dyn Sampler>>;
}

/// Pure random sampling. O(1/√N) convergence.
pub struct NaiveRandomSampler {
    rng: SmallRng,
}

impl NaiveRandomSampler {
    pub fn new() -> Self {
        Self {
            rng: SmallRng::from_rng(&mut rand::rng()),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Default for NaiveRandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for NaiveRandomSampler {
    fn get_next_1d(&mut self) -> f64 {
        self.rng.random()
    }

    fn get_next_2d(&mut self) -> [f64; 2] {
        [self.rng.random(), self.rng.random()]
    }

    fn split(&mut self, num: usize) -> Vec<Box<dyn Sampler>> {
        (0..num)
            .map(|_| Box::new(NaiveRandomSampler::with_seed(self.rng.random())) as Box<dyn Sampler>)
            .collect()
    }
}

/// Jittered grid sampling: one random sample per cell in a `sqrt_spp × sqrt_spp` grid.
/// Lower variance than naive at the same O(1/√N) convergence rate.
pub struct StratifiedRandomSampler {
    rng: SmallRng,
    sample_idx: usize,
    sqrt_spp: usize,
    sqrt_spp_inv: f64,
}

impl StratifiedRandomSampler {
    pub fn new(sqrt_spp: usize, seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            sample_idx: 0,
            sqrt_spp,
            sqrt_spp_inv: 1.0 / sqrt_spp as f64,
        }
    }

    /// Start at a given sample index — useful for progressive rendering.
    pub fn at_sample(sqrt_spp: usize, sample_idx: usize, seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            sample_idx,
            sqrt_spp,
            sqrt_spp_inv: 1.0 / sqrt_spp as f64,
        }
    }
}

impl Sampler for StratifiedRandomSampler {
    fn get_next_1d(&mut self) -> f64 {
        let cell = self.sample_idx;
        self.sample_idx += 1;
        (cell as f64 + self.rng.random::<f64>()) * self.sqrt_spp_inv
    }

    fn get_next_2d(&mut self) -> [f64; 2] {
        let si = self.sample_idx / self.sqrt_spp;
        let sj = self.sample_idx % self.sqrt_spp;
        self.sample_idx += 1;

        let px = (si as f64 + self.rng.random::<f64>()) * self.sqrt_spp_inv;
        let py = (sj as f64 + self.rng.random::<f64>()) * self.sqrt_spp_inv;
        [px, py]
    }

    fn split(&mut self, num: usize) -> Vec<Box<dyn Sampler>> {
        (0..num)
            .map(|_| {
                Box::new(StratifiedRandomSampler::new(
                    self.sqrt_spp,
                    self.rng.random(),
                )) as Box<dyn Sampler>
            })
            .collect()
    }
}

/// Sobol quasi-random sequence with random digital shift.
/// O((log N)²/N) convergence — typically 2-4× fewer samples than stratified.
///
/// 1D and 2D paths use independent index counters, so they can be mixed freely.
pub struct SobolSeqSampler {
    index_2d: u32,
    x_2d: u32,
    y_2d: u32,
    shift_x_2d: u32,
    shift_y_2d: u32,

    index_1d: u32,
    x_1d: u32,
    shift_x_1d: u32,
}

impl SobolSeqSampler {
    /// Direction numbers as left-aligned u32 fixed-point (denominator 2³²).
    /// Dim 0: Van der Corput. Dim 1: primitive polynomial x³ + x + 1.
    const SOBOL_DIRS: [[u32; 32]; 2] = [
        [
            0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000,
            0x01000000, 0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000,
            0x00020000, 0x00010000, 0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800,
            0x00000400, 0x00000200, 0x00000100, 0x00000080, 0x00000040, 0x00000020, 0x00000010,
            0x00000008, 0x00000004, 0x00000002, 0x00000001,
        ],
        [
            0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000,
            0xff000000, 0x88800000, 0xccc00000, 0xaaa00000, 0xfff00000, 0x88880000, 0xcccc0000,
            0xaaaa0000, 0xffff0000, 0x88888000, 0xccccc000, 0xaaaaa000, 0xfffff000, 0x88888800,
            0xcccccc00, 0xaaaaaa00, 0xffffff00, 0x88888880, 0xccccccc0, 0xaaaaaaa0, 0xfffffff0,
            0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
        ],
    ];

    pub fn new() -> Self {
        let mut rng = SmallRng::from_rng(&mut rand::rng());
        Self {
            index_2d: 0,
            x_2d: 0,
            y_2d: 0,
            shift_x_2d: rng.random(),
            shift_y_2d: rng.random(),
            index_1d: 0,
            x_1d: 0,
            shift_x_1d: rng.random(),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        Self {
            index_2d: 0,
            x_2d: 0,
            y_2d: 0,
            shift_x_2d: rng.random(),
            shift_y_2d: rng.random(),
            index_1d: 0,
            x_1d: 0,
            shift_x_1d: rng.random(),
        }
    }

    /// Deterministic shifts from pixel coordinates.
    pub fn for_pixel(pixel_x: i32, pixel_y: i32) -> Self {
        let mut rng = SmallRng::seed_from_u64(pixel_x as u64 * 7919 + pixel_y as u64 * 104729);
        Self {
            index_2d: 0,
            x_2d: 0,
            y_2d: 0,
            shift_x_2d: rng.random(),
            shift_y_2d: rng.random(),
            index_1d: 0,
            x_1d: 0,
            shift_x_1d: rng.random(),
        }
    }

    /// Gray code incremental: XOR the direction number for the flipped bit.
    fn advance_2d(&mut self) {
        if self.index_2d > 0 {
            let c = self.index_2d.trailing_zeros() as usize;
            self.x_2d ^= Self::SOBOL_DIRS[0][c];
            self.y_2d ^= Self::SOBOL_DIRS[1][c];
        }
        self.index_2d += 1;
    }

    fn advance_1d(&mut self) {
        if self.index_1d > 0 {
            let c = self.index_1d.trailing_zeros() as usize;
            self.x_1d ^= Self::SOBOL_DIRS[0][c];
        }
        self.index_1d += 1;
    }
}

impl Default for SobolSeqSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for SobolSeqSampler {
    fn get_next_1d(&mut self) -> f64 {
        self.advance_1d();
        let sx = self.x_1d ^ self.shift_x_1d;
        let inv = 1.0 / (1u64 << 32) as f64;
        sx as f64 * inv
    }

    fn get_next_2d(&mut self) -> [f64; 2] {
        self.advance_2d();
        let sx = self.x_2d ^ self.shift_x_2d;
        let sy = self.y_2d ^ self.shift_y_2d;
        let inv = 1.0 / (1u64 << 32) as f64;
        [sx as f64 * inv, sy as f64 * inv]
    }

    fn split(&mut self, num: usize) -> Vec<Box<dyn Sampler>> {
        (0..num)
            .map(|_| {
                Box::new(SobolSeqSampler::with_seed(
                    self.index_2d as u64 * 31337 + self.index_1d as u64 * 7919 + self.x_2d as u64,
                )) as Box<dyn Sampler>
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_produces_values_in_unit_interval() {
        let mut s = NaiveRandomSampler::new();
        for _ in 0..1000 {
            let v = s.get_next_1d();
            assert!((0.0..1.0).contains(&v), "1d out of range: {v}");
            let [x, y] = s.get_next_2d();
            assert!((0.0..1.0).contains(&x), "2d x out of range: {x}");
            assert!((0.0..1.0).contains(&y), "2d y out of range: {y}");
        }
    }

    #[test]
    fn stratified_produces_values_in_unit_interval() {
        let mut s = StratifiedRandomSampler::new(16, 42);
        for _ in 0..256 {
            let [x, y] = s.get_next_2d();
            assert!((0.0..1.0).contains(&x), "2d x out of range: {x}");
            assert!((0.0..1.0).contains(&y), "2d y out of range: {y}");
        }
    }

    #[test]
    fn stratified_covers_all_cells() {
        let mut s = StratifiedRandomSampler::new(4, 42);
        let mut cells = [[false; 4]; 4];
        for _ in 0..16 {
            let [x, y] = s.get_next_2d();
            let ci = (x * 4.0) as usize;
            let cj = (y * 4.0) as usize;
            cells[ci][cj] = true;
        }
        assert!(cells.iter().all(|row| row.iter().all(|&c| c)));
    }

    #[test]
    fn sobol_2d_produces_values_in_unit_interval() {
        let mut s = SobolSeqSampler::new();
        for _ in 0..256 {
            let [x, y] = s.get_next_2d();
            assert!((0.0..1.0).contains(&x), "2d x out of range: {x}");
            assert!((0.0..1.0).contains(&y), "2d y out of range: {y}");
        }
    }

    #[test]
    fn sobol_1d_produces_values_in_unit_interval() {
        let mut s = SobolSeqSampler::new();
        for _ in 0..256 {
            let v = s.get_next_1d();
            assert!((0.0..1.0).contains(&v), "1d out of range: {v}");
        }
    }

    #[test]
    fn sobol_1d_and_2d_are_independent() {
        let mut s = SobolSeqSampler::new();
        let d2_samples: Vec<_> = (0..10).map(|_| s.get_next_2d()).collect();
        let d1_samples: Vec<_> = (0..10).map(|_| s.get_next_1d()).collect();
        let d2_x: Vec<f64> = d2_samples.iter().map(|[x, _]| *x).collect();
        assert_ne!(d1_samples, d2_x);
    }

    #[test]
    fn sobol_first_sample_is_shift() {
        let mut rng = SmallRng::seed_from_u64(42);
        let shift_x: u32 = rng.random();
        let shift_y: u32 = rng.random();

        let mut s = SobolSeqSampler {
            index_2d: 0,
            x_2d: 0,
            y_2d: 0,
            shift_x_2d: shift_x,
            shift_y_2d: shift_y,
            index_1d: 0,
            x_1d: 0,
            shift_x_1d: 0,
        };

        let [x, y] = s.get_next_2d();
        let inv = 1.0 / (1u64 << 32) as f64;
        assert!((x - shift_x as f64 * inv).abs() < 1e-15);
        assert!((y - shift_y as f64 * inv).abs() < 1e-15);
    }

    #[test]
    fn sobol_split_produces_independent_samplers() {
        let mut s = SobolSeqSampler::new();
        let children = s.split(3);
        assert_eq!(children.len(), 3);
        let parent_sample = s.get_next_2d();
        assert!((0.0..1.0).contains(&parent_sample[0]));
        for mut child in children {
            let child_sample = child.get_next_2d();
            assert!((0.0..1.0).contains(&child_sample[0]));
        }
    }

    #[test]
    fn naive_split_produces_independent_samplers() {
        let mut s = NaiveRandomSampler::new();
        let children = s.split(3);
        assert_eq!(children.len(), 3);
        let parent_sample = s.get_next_1d();
        assert!((0.0..1.0).contains(&parent_sample));
        for mut child in children {
            let child_sample = child.get_next_1d();
            assert!((0.0..1.0).contains(&child_sample));
        }
    }

    #[test]
    fn stratified_split_preserves_sqrt_spp() {
        let mut s = StratifiedRandomSampler::new(16, 42);
        let mut children = s.split(2);
        assert_eq!(children.len(), 2);
        for child in &mut children {
            for _ in 0..256 {
                let [x, y] = child.get_next_2d();
                assert!((0.0..1.0).contains(&x));
                assert!((0.0..1.0).contains(&y));
            }
        }
    }
}
