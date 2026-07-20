use crate::shape::Region2D;

/// Region type for an arbitrary convex or simple polygon defined by vertices
/// in (a, b) parametric space.
///
/// Vertices should be supplied in order (clockwise or counter-clockwise).
/// `bbox` is the polygon's axis-aligned bounding box, used for rejection sampling.
#[derive(Clone)]
pub struct PolygonRegion {
    vertices: Vec<(f32, f32)>,
    bbox: (f32, f32, f32, f32), // (a_min, a_max, b_min, b_max)
}

impl PolygonRegion {
    /// Build a polygon from vertices in (a, b) space. The bounding box is
    /// computed automatically from the vertices.
    pub fn new(vertices: Vec<(f32, f32)>) -> Self {
        let mut a_min = f32::INFINITY;
        let mut a_max = f32::NEG_INFINITY;
        let mut b_min = f32::INFINITY;
        let mut b_max = f32::NEG_INFINITY;
        for &(a, b) in &vertices {
            if a < a_min {
                a_min = a;
            }
            if a > a_max {
                a_max = a;
            }
            if b < b_min {
                b_min = b;
            }
            if b > b_max {
                b_max = b;
            }
        }
        Self {
            vertices,
            bbox: (a_min, a_max, b_min, b_max),
        }
    }
}

impl Region2D for PolygonRegion {
    fn contains(&self, a: f32, b: f32) -> bool {
        // Standard ray-casting point-in-polygon. O(n) per check.
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        for i in 0..n {
            let (xi, yi) = self.vertices[i];
            let (xj, yj) = self.vertices[(i + 1) % n];
            // Edge crosses the horizontal ray at y = b, and the point is
            // to the left of the intersection x.
            if (yi > b) != (yj > b) {
                let x_intersect = (xj - xi) * (b - yi) / (yj - yi) + xi;
                if a < x_intersect {
                    inside = !inside;
                }
            }
        }
        inside
    }

    fn area(&self) -> f32 {
        // Shoelace formula.
        let n = self.vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..n {
            let (xi, yi) = self.vertices[i];
            let (xj, yj) = self.vertices[(i + 1) % n];
            sum += xi * yj - xj * yi;
        }
        0.5 * sum.abs()
    }

    fn bounding_box_area(&self) -> f32 {
        let (a_min, a_max, b_min, b_max) = self.bbox;
        (a_max - a_min) * (b_max - b_min)
    }

    fn sample(&self, u: f32, v: f32) -> (f32, f32) {
        let (a_min, a_max, b_min, b_max) = self.bbox;
        let mut u = u;
        let mut v = v;
        for _ in 0..32 {
            let a = u * (a_max - a_min) + a_min;
            let b = v * (b_max - b_min) + b_min;
            if self.contains(a, b) {
                return (a, b);
            }
            u = (u + 0.618_034).fract();
            v = (v + 0.618_034).fract();
        }
        // Fallback: centroid of bounding box
        ((a_min + a_max) * 0.5, (b_min + b_max) * 0.5)
    }
}
