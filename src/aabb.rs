use crate::interval::Interval;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Axis-aligned bounding box used for broad-phase ray culling and BVH traversal.
#[derive(Default, Debug, Clone, Copy)]
pub struct Aabb {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
}

impl Aabb {
    const DELTA: f64 = 0.0001;
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
        let new = Self {
            x: Interval::from(a[0].min(b[0]), a[0].max(b[0])),
            y: Interval::from(a[1].min(b[1]), a[1].max(b[1])),
            z: Interval::from(a[2].min(b[2]), a[2].max(b[2])),
        };
        new.pad_to_minimums()
    }

    fn pad_to_minimums(mut self) -> Self {
        if self.x.size() < Self::DELTA {
            self.x.expand(Self::DELTA);
        }
        if self.y.size() < Self::DELTA {
            self.y.expand(Self::DELTA);
        }
        if self.z.size() < Self::DELTA {
            self.z.expand(Self::DELTA);
        }
        self
    }

    /// Returns the union of two AABBs.
    pub fn merge(&self, other: Aabb) -> Self {
        Self {
            x: Interval::from_intervals(&self.x, &other.x),
            y: Interval::from_intervals(&self.y, &other.y),
            z: Interval::from_intervals(&self.z, &other.z),
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

    pub fn centroid(&self) -> Point3 {
        Point3::from(
            0.5 * (self.x.min + self.x.max),
            0.5 * (self.y.min + self.y.max),
            0.5 * (self.z.min + self.z.max),
        )
    }

    pub fn translate(mut self, offset: Vec3) -> Self {
        self.x.min += offset.x;
        self.x.max += offset.x;
        self.y.min += offset.y;
        self.y.max += offset.y;
        self.z.min += offset.z;
        self.z.max += offset.z;

        self
    }

    /// Ray-box test using the slab method.
    ///
    /// `ray_t` is narrowed per-axis and early-outs once the interval collapses.
    pub fn hit(&self, ray: &Ray, mut ray_t: Interval) -> bool {
        // slab method - test intersection against each axis pair
        for axis in 0..3 {
            let ax = self.axis_interval(axis);
            let inv_d = 1.0 / ray.direction[axis as usize];
            let mut t0 = (ax.min - ray.origin[axis as usize]) * inv_d;
            let mut t1 = (ax.max - ray.origin[axis as usize]) * inv_d;

            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }

            ray_t.min = ray_t.min.max(t0);
            ray_t.max = ray_t.max.min(t1);

            if ray_t.max <= ray_t.min {
                return false;
            }
        }

        true
    }
}
