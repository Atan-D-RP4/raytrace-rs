use crate::vec3::Color3;

pub struct FilmTile {
    /// Bounds of the tile in pixel coordinates (x_min, x_max, y_min, y_max)
    pub bounds: [u32; 4],
    /// Cached width = bounds[1] - bounds[0]
    pub width: u32,
    /// Accumulated color for each pixel in the tile
    pub pixels: Vec<Color3>,
    /// A parallel vector to `pixels` that tracks whether each pixel has been sampled at least once.
    pub sampled: Vec<bool>,
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
            sampled: vec![false; (width * height) as usize],
        }
    }

    /// Adds a sample color to the pixel at (x, y) within the tile.
    /// The color is accumulated in linear space.
    pub fn add_sample(&mut self, x: u32, y: u32, color: Color3) {
        let x_min = self.bounds[0];
        let y_min = self.bounds[2];
        let index = ((y - y_min) * self.width + (x - x_min)) as usize;
        self.pixels[index] += color;
        self.sampled[index] = true;
    }
}
