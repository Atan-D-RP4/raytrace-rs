use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::Point3;

/// Axis-aligned bounding box used for broad-phase ray culling and BVH traversal.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
}

impl Aabb {
    /// Creates an empty bounding box.
    pub fn new() -> Self {
        Self {
            x: Interval::EMPTY,
            y: Interval::EMPTY,
            z: Interval::EMPTY,
        }
    }

    /// Creates a bounding box directly from axis intervals.
    pub fn from_intervals(x: Interval, y: Interval, z: Interval) -> Self {
        Self { x, y, z }
    }

    /// Creates a bounding box that encloses both points.
    ///
    /// Treat the two points a and b as extrema for the bounding box, so we don't require a
    /// particular minimum/maximum coordinate order.
    pub fn from_points(a: &Point3, b: &Point3) -> Self {
        Self {
            x: if a[0] <= b[0] {
                Interval::from(a[0], b[0])
            } else {
                Interval::from(b[0], a[0])
            },
            y: if a[1] <= b[1] {
                Interval::from(a[1], b[1])
            } else {
                Interval::from(b[1], a[1])
            },
            z: if a[2] <= b[2] {
                Interval::from(a[2], b[2])
            } else {
                Interval::from(b[2], a[2])
            },
        }
    }

    /// Returns the union of two AABBs.
    pub fn merge(a: Aabb, b: Aabb) -> Self {
        Self {
            x: Interval::from_intervals(&a.x, &b.x),
            y: Interval::from_intervals(&a.y, &b.y),
            z: Interval::from_intervals(&a.z, &b.z),
        }
    }

    /// Returns the interval for the selected axis (0=x, 1=y, 2=z).
    pub fn axis_interval(&self, axis: i32) -> &Interval {
        match axis {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Invalid axis index: {}", axis),
        }
    }

    /// Returns the axis index with the largest extent.
    pub fn longest_axis(&self) -> i32 {
        let (x, y, z) = (self.x.size(), self.y.size(), self.z.size());
        if x > y && x > z {
            0
        } else if y > z {
            1
        } else {
            2
        }
    }

    /// Ray-box test using the slab method.
    ///
    /// `ray_t` is narrowed per-axis and early-outs once the interval collapses.
    pub fn hit(&self, ray: &Ray, mut ray_t: Interval) -> bool {
        // slab method - test intersection against each axis pair
        (0..3).all(|axis| {
            let ax = self.axis_interval(axis);
            let adinv = 1.0 / ray.direction[axis as usize];
            let t0 = (ax.min - ray.origin[axis as usize]) * adinv;
            let t1 = (ax.max - ray.origin[axis as usize]) * adinv;

            if t0 < t1 {
                if t0 > ray_t.min {
                    ray_t.min = t0
                }
                if t1 < ray_t.max {
                    ray_t.max = t1
                }
            } else {
                if t1 > ray_t.min {
                    ray_t.min = t1
                }
                if t0 < ray_t.max {
                    ray_t.max = t0
                }
            }

            ray_t.max > ray_t.min
        })
    }
}
