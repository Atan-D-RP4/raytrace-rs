use std::simd::num::SimdFloat;
use std::simd::prelude::*;
use std::simd::{Mask, Simd};

use glam::Vec3;

use crate::math::interval::Interval;
use crate::math::vec3::Point3;
use crate::ray::Ray;

pub type Aabb = AabbPacked<1>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AabbPacked<const W: usize> {
    /// The minimum values for each axis, packed into a SIMD-friendly structure.
    pub min: [[f32; W]; 3],
    /// The maximum values for each axis, packed into a SIMD-friendly structure.
    pub max: [[f32; W]; 3],
}

impl<const W: usize> AabbPacked<W> {
    /// A small delta to pad the bounding box when its size is too small.
    const DELTA: f32 = 0.0001;

    /// Create a new AABB from the given min and max arrays.
    #[inline]
    pub const fn new(min: [[f32; W]; 3], max: [[f32; W]; 3]) -> Self {
        Self { min, max }
    }

    /// Create an AABB from three arrays of intervals, one for each axis.
    #[inline]
    pub fn from_intervals(x: &[Interval; W], y: &[Interval; W], z: &[Interval; W]) -> Self {
        let mut min = [[0.0; W]; 3];
        let mut max = [[0.0; W]; 3];

        for i in 0..W {
            min[0][i] = x[i].min;
            min[1][i] = y[i].min;
            min[2][i] = z[i].min;

            max[0][i] = x[i].max;
            max[1][i] = y[i].max;
            max[2][i] = z[i].max;
        }

        Self { min, max }
    }

    /// Create an AABB that tightly bounds the given array of points.
    #[inline]
    pub fn from_points(points: &[Point3; W]) -> Self {
        let mut min = [[0.0; W]; 3];
        let mut max = [[0.0; W]; 3];

        for i in 0..W {
            let p = points[i].into_inner();
            min[0][i] = p[0];
            min[1][i] = p[1];
            min[2][i] = p[2];

            max[0][i] = p[0];
            max[1][i] = p[1];
            max[2][i] = p[2];
        }

        Self { min, max }
    }

    /// Merge two AABBs, padding the result to ensure a minimum size.
    #[inline]
    pub fn merge(&self, other: &Self) -> Self {
        let mut merged = self.merge_unpadded(other);
        merged.pad_to_minimums();
        merged
    }

    /// Merge two AABBs without padding the result. This is useful for intermediate calculations
    /// where padding is not desired.
    #[inline]
    pub fn merge_unpadded(&self, other: &Self) -> Self {
        let mut new = Self::empty();
        for axis in 0..3 {
            let (self_min, self_max) = self.axis(axis);
            let (other_min, other_max) = other.axis(axis);
            let (out_min, out_max) = new.axis_mut(axis);
            for i in 0..W {
                out_min[i] = self_min[i].min(other_min[i]);
                out_max[i] = self_max[i].max(other_max[i]);
            }
        }
        new
    }

    /// Creates an empty bounding box with min = +∞ and max = -∞.
    pub fn empty() -> Self {
        Self {
            min: [[f32::INFINITY; W]; 3],
            max: [[f32::NEG_INFINITY; W]; 3],
        }
    }

    /// Extracts the AABB for a specific child index.
    ///
    /// Returns (min, max) where each is [f32; 3] representing the AABB for child `index`.
    pub fn child_aabb(&self, index: usize) -> ([f32; 3], [f32; 3]) {
        let min = [self.min[0][index], self.min[1][index], self.min[2][index]];
        let max = [self.max[0][index], self.max[1][index], self.max[2][index]];
        (min, max)
    }

    /// Returns the min and max arrays for a specific axis (0 = x, 1 = y, 2 = z).
    #[inline]
    pub fn axis(&self, axis: usize) -> (&[f32; W], &[f32; W]) {
        (&self.min[axis], &self.max[axis])
    }

    /// Returns mutable references to the min and max arrays for a specific axis (0 = x, 1 = y, 2 =
    /// z).
    #[inline]
    pub fn axis_mut(&mut self, axis: usize) -> (&mut [f32; W], &mut [f32; W]) {
        (&mut self.min[axis], &mut self.max[axis])
    }

    /// Returns the index of the axis with the longest extent (0 = x, 1 = y, 2 = z).
    #[inline]
    pub fn longest_axis(&self) -> usize {
        let mut longest = 0;
        let mut max_extent = 0.0;

        for axis in 0..3 {
            let extent = self.max[axis][0] - self.min[axis][0];
            if extent > max_extent {
                max_extent = extent;
                longest = axis;
            }
        }

        longest
    }

    /// Returns the surface area of each AABB in the packed structure.
    #[inline]
    pub fn surface_area(&self) -> [f32; W] {
        let mut areas = [0.0; W];

        areas.iter_mut().enumerate().for_each(|(i, area)| {
            let dx = self.max[0][i] - self.min[0][i];
            let dy = self.max[1][i] - self.min[1][i];
            let dz = self.max[2][i] - self.min[2][i];
            *area = 2.0 * (dx * dy + dx * dz + dy * dz);
        });

        areas
    }

    /// Returns the centroid of each AABB in the packed structure.
    #[inline]
    pub fn centroid(&self) -> [[f32; W]; 3] {
        let mut centroids = [[0.0; W]; 3];

        centroids
            .iter_mut()
            .enumerate()
            .for_each(|(axis, centroid)| {
                centroid.iter_mut().enumerate().for_each(|(i, v)| {
                    *v = 0.5 * (self.min[axis][i] + self.max[axis][i]);
                });
            });

        centroids
    }

    /// Translates all AABBs in the packed structure by the given offset vector.
    #[inline]
    pub fn translate(mut self, offset: Vec3) -> Self {
        for axis in 0..3 {
            for i in 0..W {
                self.min[axis][i] += offset[axis];
                self.max[axis][i] += offset[axis];
            }
        }
        self
    }

    /// Branchless slab test against all W children for a batch of rays.
    ///
    /// `ray_origins` and `ray_directions` are arrays of length W, where each element is a 3D vector
    /// representing the origin and direction of a ray. `ray_t` is an array of length W, where each
    /// element is a 2D vector representing the minimum and maximum t values for the ray segment.
    #[inline]
    pub fn hit(
        &self,
        ray_origins: &[[f32; 3]; W],
        ray_directions: &[[f32; 3]; W],
        ray_t: &mut [[f32; 2]; W],
    ) -> [bool; W] {
        let mut hits = [true; W];

        for axis in 0..3 {
            let (min_vals, max_vals) = self.axis(axis);
            for i in 0..W {
                let inv_d = 1.0 / ray_directions[i][axis];
                let t0 = (min_vals[i] - ray_origins[i][axis]) * inv_d;
                let t1 = (max_vals[i] - ray_origins[i][axis]) * inv_d;

                let t_min = t0.min(t1);
                let t_max = t0.max(t1);

                ray_t[i][0] = ray_t[i][0].max(t_min);
                ray_t[i][1] = ray_t[i][1].min(t_max);

                if ray_t[i][1] <= ray_t[i][0] {
                    hits[i] = false;
                }
            }
        }

        hits
    }

    #[inline]
    fn pad_to_minimums(&mut self) {
        for axis in 0..3 {
            for i in 0..W {
                let size = self.max[axis][i] - self.min[axis][i];
                if size.is_finite() && size < Self::DELTA {
                    let mid = 0.5 * (self.min[axis][i] + self.max[axis][i]);
                    self.min[axis][i] = mid - 0.5 * Self::DELTA;
                    self.max[axis][i] = mid + 0.5 * Self::DELTA;
                }
            }
        }
    }

    /// Branchless slab test against all W children for a single ray.
    ///
    /// Returns a u16 bitmask where bit `i` is set if the ray segment
    /// `[tmin, tmax]` intersects child i's AABB. Uses explicit `std::simd`
    /// for guaranteed packed AABB testing.
    #[inline]
    pub fn hit_mask(&self, ray: &Ray, tmin: f32, tmax: f32) -> u16
    where
        Simd<f32, W>: SimdPartialOrd + SimdFloat,
        <Simd<f32, W> as SimdPartialEq>::Mask: Into<Mask<i32, W>>,
    {
        let ox = Simd::<f32, W>::splat(ray.origin.x());
        let oy = Simd::<f32, W>::splat(ray.origin.y());
        let oz = Simd::<f32, W>::splat(ray.origin.z());
        let idx = Simd::<f32, W>::splat(ray.inverse_direction.x());
        let idy = Simd::<f32, W>::splat(ray.inverse_direction.y());
        let idz = Simd::<f32, W>::splat(ray.inverse_direction.z());

        let mut lo = Simd::<f32, W>::splat(tmin);
        let mut hi = Simd::<f32, W>::splat(tmax);

        // X axis
        let min_x = Simd::from_array(self.min[0]);
        let max_x = Simd::from_array(self.max[0]);
        let t0 = (min_x - ox) * idx;
        let t1 = (max_x - ox) * idx;
        lo = lo.simd_max(t0.simd_min(t1));
        hi = hi.simd_min(t0.simd_max(t1));

        // Y axis
        let min_y = Simd::from_array(self.min[1]);
        let max_y = Simd::from_array(self.max[1]);
        let t0 = (min_y - oy) * idy;
        let t1 = (max_y - oy) * idy;
        lo = lo.simd_max(t0.simd_min(t1));
        hi = hi.simd_min(t0.simd_max(t1));

        // Z axis
        let min_z = Simd::from_array(self.min[2]);
        let max_z = Simd::from_array(self.max[2]);
        let t0 = (min_z - oz) * idz;
        let t1 = (max_z - oz) * idz;
        lo = lo.simd_max(t0.simd_min(t1));
        hi = hi.simd_min(t0.simd_max(t1));

        // Compare and extract bitmask.
        let mask: Mask<i32, W> = hi.simd_gt(lo).into();
        mask.to_bitmask() as u16
    }
}

// ---------------------------------------------------------------------------
// From impls for AabbPacked
// ---------------------------------------------------------------------------

/// Slot-0 scalar helpers for use by builder code that operates on Aabb = AabbPacked<1>.
impl<const W: usize> AabbPacked<W> {
    /// Centroid of slot 0 (the first AABB).
    pub fn centroid_point(&self) -> Point3 {
        Point3::new(
            0.5 * (self.min[0][0] + self.max[0][0]),
            0.5 * (self.min[1][0] + self.max[1][0]),
            0.5 * (self.min[2][0] + self.max[2][0]),
        )
    }

    /// Create an AABB from two corner points. Only fills slot 0; other slots
    /// are left empty (always-miss for slab tests).
    pub fn from_corners(p1: Point3, p2: Point3) -> Self {
        let mut result = Self::empty();

        let b_min = p1.min(p2.into_inner()).into_inner();
        let b_max = p1.max(p2.into_inner()).into_inner();
        result.min = b_min.to_array().map(|v| [v; W]);
        result.max = b_max.to_array().map(|v| [v; W]);

        result
    }

    /// Single-ray AABB hit test against slot 0. Returns true if the ray segment
    /// [ray_t.min, ray_t.max] intersects slot 0's AABB.
    pub fn hit_single(&self, ray: &Ray, ray_t: &Interval) -> bool {
        // Inline slab test for slot 0.
        let ox = ray.origin.x();
        let oy = ray.origin.y();
        let oz = ray.origin.z();
        let idx = ray.inverse_direction.x();
        let idy = ray.inverse_direction.y();
        let idz = ray.inverse_direction.z();
        let tmin = ray_t.min;
        let tmax = ray_t.max;

        let mut lo = tmin;
        let mut hi = tmax;

        // X slab.
        let t0 = (self.min[0][0] - ox) * idx;
        let t1 = (self.max[0][0] - ox) * idx;
        lo = lo.max(t0.min(t1));
        hi = hi.min(t0.max(t1));
        if hi <= lo {
            return false;
        }

        // Y slab.
        let t0 = (self.min[1][0] - oy) * idy;
        let t1 = (self.max[1][0] - oy) * idy;
        lo = lo.max(t0.min(t1));
        hi = hi.min(t0.max(t1));
        if hi <= lo {
            return false;
        }

        // Z slab.
        let t0 = (self.min[2][0] - oz) * idz;
        let t1 = (self.max[2][0] - oz) * idz;
        lo = lo.max(t0.min(t1));
        hi = hi.min(t0.max(t1));
        hi > lo
    }
}

// ---------------------------------------------------------------------------
// From impls for AabbPacked
// ---------------------------------------------------------------------------

/// Broadcast a single scalar Aabb (ref) to all W slots.
impl<const W: usize> From<&Aabb> for AabbPacked<W> {
    fn from(aabb: &Aabb) -> Self {
        Self {
            min: [
                [aabb.min[0][0]; W],
                [aabb.min[1][0]; W],
                [aabb.min[2][0]; W],
            ],
            max: [
                [aabb.max[0][0]; W],
                [aabb.max[1][0]; W],
                [aabb.max[2][0]; W],
            ],
        }
    }
}

/// Pack up to W scalar Aabbs from an iterator. Remaining slots are left empty
/// (min = +∞, max = -∞) so they always miss slab tests.
impl<const W: usize> FromIterator<Aabb> for AabbPacked<W> {
    fn from_iter<I: IntoIterator<Item = Aabb>>(iter: I) -> Self {
        let mut result = Self::empty();
        for (i, aabb) in iter.into_iter().enumerate() {
            if i >= W {
                break;
            }
            result.min[0][i] = aabb.min[0][0];
            result.min[1][i] = aabb.min[1][0];
            result.min[2][i] = aabb.min[2][0];
            result.max[0][i] = aabb.max[0][0];
            result.max[1][i] = aabb.max[1][0];
            result.max[2][i] = aabb.max[2][0];
        }
        result
    }
}

/// Unpack all W children as scalar Aabbs.
impl<const W: usize> From<&AabbPacked<W>> for [Aabb; W] {
    fn from(packed: &AabbPacked<W>) -> Self {
        core::array::from_fn(|i| {
            Aabb::from_intervals(
                &[Interval::from(packed.min[0][i], packed.max[0][i])],
                &[Interval::from(packed.min[1][i], packed.max[1][i])],
                &[Interval::from(packed.min[2][i], packed.max[2][i])],
            )
        })
    }
}

/// Owned version.
impl<const W: usize> From<AabbPacked<W>> for [Aabb; W] {
    fn from(packed: AabbPacked<W>) -> Self {
        (&packed).into()
    }
}

// ---------------------------------------------------------------------------
// Split & join for AabbPacked (W*2 ↔ W)
// ---------------------------------------------------------------------------
//
// Since const-generic arithmetic (W/2, W*2) in return types needs unstable
// `generic_const_exprs`, we generate concrete impls with a macro.
// ---------------------------------------------------------------------------

macro_rules! impl_aabb_split_join {
    ($w:expr, $half:expr) => {
        impl AabbPacked<$w> {
            /// Split into lo [0..$half) and hi [$half..$w).
            pub fn split_half(&self) -> (AabbPacked<$half>, AabbPacked<$half>) {
                let lo = AabbPacked {
                    min: core::array::from_fn(|axis| core::array::from_fn(|i| self.min[axis][i])),
                    max: core::array::from_fn(|axis| core::array::from_fn(|i| self.max[axis][i])),
                };
                let hi = AabbPacked {
                    min: core::array::from_fn(|axis| {
                        core::array::from_fn(|i| self.min[axis][i + $half])
                    }),
                    max: core::array::from_fn(|axis| {
                        core::array::from_fn(|i| self.max[axis][i + $half])
                    }),
                };
                (lo, hi)
            }
        }

        impl From<(AabbPacked<$half>, AabbPacked<$half>)> for AabbPacked<$w> {
            fn from(halves: (AabbPacked<$half>, AabbPacked<$half>)) -> Self {
                let (lo, hi) = halves;
                AabbPacked {
                    min: core::array::from_fn(|axis| {
                        let mut arr = [0.0f32; $w];
                        let mut i = 0;
                        while i < $half {
                            arr[i] = lo.min[axis][i];
                            arr[i + $half] = hi.min[axis][i];
                            i += 1;
                        }
                        arr
                    }),
                    max: core::array::from_fn(|axis| {
                        let mut arr = [0.0f32; $w];
                        let mut i = 0;
                        while i < $half {
                            arr[i] = lo.max[axis][i];
                            arr[i + $half] = hi.max[axis][i];
                            i += 1;
                        }
                        arr
                    }),
                }
            }
        }
    };
}

impl_aabb_split_join!(4, 2);
impl_aabb_split_join!(8, 4);
impl_aabb_split_join!(16, 8);
