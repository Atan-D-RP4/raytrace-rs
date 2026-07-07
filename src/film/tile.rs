use crate::vec3::Color3;

pub struct FilmTile {
    /// Bounds of the tile in pixel coordinates (x_min, x_max, y_min, y_max)
    pub bounds: [u32; 4],
    /// Cached width = bounds[1] - bounds[0]
    pub width: u32,
    /// Accumulated weighted color for each pixel in the tile: sum(color * weight)
    pub pixels: Vec<Color3>,
    /// Sum of raw (unweighted) sample colors for variance estimation.
    pub raw_sum: Vec<Color3>,
    /// A parallel vector to `pixels` that tracks whether each pixel has been sampled at least once.
    pub sampled: Vec<bool>,
    /// Running sum of sample weights for each pixel (used by reconstruction filters).
    pub weight_sum: Vec<f64>,
    /// Actual number of samples per pixel (independent of filter weights).
    /// Used by the convergence system to determine if enough samples have been taken.
    pub sample_count: Vec<u32>,
    /// Whether this tile was zeroed this pass (has unconverged pixels).
    /// Tiles where all pixels are converged skip zeroing and merging.
    pub dirty: bool,
}

impl FilmTile {
    /// Creates a new FilmTile with the given bounds. The pixel buffer is initialized to zero.
    pub fn new(bounds: [u32; 4]) -> Self {
        let width = bounds[1] - bounds[0];
        let height = bounds[3] - bounds[2];
        Self {
            bounds,
            width,
            pixels: vec![Color3::ZERO; (width * height) as usize],
            raw_sum: vec![Color3::ZERO; (width * height) as usize],
            sampled: vec![false; (width * height) as usize],
            weight_sum: vec![0.0; (width * height) as usize],
            sample_count: vec![0; (width * height) as usize],
            dirty: true,
        }
    }

    /// Adds a sample color to the pixel at (x, y) within the tile with equal weight.
    /// The color is accumulated in linear space.
    pub fn add_sample(&mut self, x: u32, y: u32, color: Color3) {
        self.add_sample_weighted(x, y, color, 1.0);
    }

    /// Adds a weighted sample color to the pixel at (x, y) within the tile.
    ///
    /// `weight` is the reconstruction filter weight (e.g., tent filter value).
    /// The color is multiplied by the weight before accumulation, and the weight
    /// is tracked separately for proper normalization during merge.
    pub fn add_sample_weighted(&mut self, x: u32, y: u32, color: Color3, weight: f64) {
        let x_min = self.bounds[0];
        let y_min = self.bounds[2];
        let index = ((y - y_min) * self.width + (x - x_min)) as usize;
        self.pixels[index] += color * weight;
        self.raw_sum[index] += color;
        self.weight_sum[index] += weight;
        self.sample_count[index] += 1;
        self.sampled[index] = true;
    }
}
