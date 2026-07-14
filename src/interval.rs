#[derive(Debug, Clone, Copy)]
pub struct Interval {
    pub min: f32,
    pub max: f32,
}

impl Default for Interval {
    fn default() -> Self {
        Self::new()
    }
}

impl Interval {
    pub const EMPTY: Interval = Interval {
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
    };

    pub const UNIVERSE: Interval = Interval {
        min: f32::NEG_INFINITY,
        max: f32::INFINITY,
    };

    pub fn new() -> Self {
        Self::EMPTY
    }

    pub fn from_intervals(a: &Interval, b: &Interval) -> Self {
        let min = a.min.min(b.min);
        let max = a.max.max(b.max);
        Self { min, max }
    }

    pub const fn from(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    pub fn size(&self) -> f32 {
        if self.max > self.min {
            self.max - self.min
        } else {
            0.0
        }
    }

    pub fn contains(&self, value: f32) -> bool {
        self.min <= value && value <= self.max
    }

    pub fn surrounds(&self, value: f32) -> bool {
        self.min < value && value < self.max
    }

    pub fn clamp(&self, value: f32) -> f32 {
        if value < self.min {
            self.min
        } else if value > self.max {
            self.max
        } else {
            value
        }
    }

    pub fn expand(&mut self, delta: f32) {
        let padding = delta / 2.0;
        self.min -= padding;
        self.max += padding;
    }
}
