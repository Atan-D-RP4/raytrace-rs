/// A packed interval: N (min, max) pairs in lane-major SoA form.
///
/// `data[0]` holds all N minimums and `data[1]` all N maximums, so a SIMD
/// slab test can load `Simd::from_array(ray_t.min())` in one register load,
/// mirroring the axis-major layout of `RayPacked<N>` and `AabbPacked`.
#[derive(Debug, Clone, Copy)]
pub struct Interval<const N: usize> {
    /// data[0] = all N mins, data[1] = all N maxs.
    data: [[f32; N]; 2],
}

impl<const N: usize> Interval<N> {
    /// Empty interval: min = +inf, max = -inf (all lanes).
    pub const EMPTY: Interval<N> = Interval {
        data: [[f32::INFINITY; N], [f32::NEG_INFINITY; N]],
    };

    /// Universe interval: min = -inf, max = +inf (all lanes).
    pub const UNIVERSE: Interval<N> = Interval {
        data: [[f32::NEG_INFINITY; N], [f32::INFINITY; N]],
    };

    #[inline]
    pub const fn new() -> Self {
        Self::EMPTY
    }

    /// Broadcasts a scalar interval to all N lanes.
    #[inline]
    pub const fn from(min: f32, max: f32) -> Self {
        Self {
            data: [[min; N], [max; N]],
        }
    }

    /// Builds a packed interval from per-lane minimums and maximums.
    #[inline]
    pub const fn from_array(min: [f32; N], max: [f32; N]) -> Self {
        Self { data: [min, max] }
    }

    /// All N minimums.
    #[inline]
    pub const fn min(&self) -> [f32; N] {
        self.data[0]
    }

    /// All N maximums.
    #[inline]
    pub const fn max(&self) -> [f32; N] {
        self.data[1]
    }

    /// Extracts lane `i` as a scalar interval.
    #[inline]
    pub const fn lane(&self, i: usize) -> Interval<1> {
        Interval {
            data: [[self.data[0][i]], [self.data[1][i]]],
        }
    }

    /// Lane-0 minimum (scalar consumers).
    #[inline]
    pub const fn min_value(&self) -> f32 {
        self.data[0][0]
    }

    /// Lane-0 maximum (scalar consumers).
    #[inline]
    pub const fn max_value(&self) -> f32 {
        self.data[1][0]
    }

    /// Lane-0 contains check.
    #[inline]
    pub const fn contains_value(&self, value: f32) -> bool {
        self.data[0][0] <= value && value <= self.data[1][0]
    }

    /// Lane-0 clamp.
    #[inline]
    pub const fn clamp_value(&self, value: f32) -> f32 {
        if value < self.data[0][0] {
            self.data[0][0]
        } else if value > self.data[1][0] {
            self.data[1][0]
        } else {
            value
        }
    }

    /// Lane-0 size.
    #[inline]
    pub const fn size_value(&self) -> f32 {
        if self.data[1][0] > self.data[0][0] {
            self.data[1][0] - self.data[0][0]
        } else {
            0.0
        }
    }
}

impl<const N: usize> Default for Interval<N> {
    fn default() -> Self {
        Self::new()
    }
}
