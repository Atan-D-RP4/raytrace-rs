use glam::Vec3;
use rand::RngExt;

use crate::math::vec3::Point3;

/// Trilinear interpolation of a 2x2x2 grid of values `c` at the point `(u, v, w)` in [0, 1]^3.
/// Legacy implementation, not used in the current codebase. Kept for reference.
#[allow(dead_code)]
fn trilinear_interp(c: [[[f32; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
    (0..2)
        .flat_map(|i| (0..2).flat_map(move |j| (0..2).map(move |k| (i, j, k))))
        .fold(0.0, |acc, (i, j, k)| {
            acc + (i as f32 * u + (1 - i) as f32 * (1.0 - u))
                * (j as f32 * v + (1 - j) as f32 * (1.0 - v))
                * (k as f32 * w + (1 - k) as f32 * (1.0 - w))
                * c[i][j][k]
        })
}

/// Perlin interpolation of a 2x2x2 grid of gradient vectors `c` at the point `(u, v, w)` in [0, 1]^3.
pub fn perlin_interp(c: [[[Vec3; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
    // Hermite smoothing
    let u = u * u * (3.0 - 2.0 * u);
    let v = v * v * (3.0 - 2.0 * v);
    let w = w * w * (3.0 - 2.0 * w);

    let mut accum = 0.0;
    // Using for-loop instead of iterators for better optimization
    for (i, c_i) in c.iter().enumerate() {
        let fu = if i == 1 { u } else { 1.0 - u };
        for (j, c_ij) in c_i.iter().enumerate() {
            let fv = if j == 1 { v } else { 1.0 - v };
            for (k, c_ijk) in c_ij.iter().enumerate() {
                let fw = if k == 1 { w } else { 1.0 - w };
                let weight = Vec3::new(u - i as f32, v - j as f32, w - k as f32);
                accum += fu * fv * fw * c_ijk.dot(weight);
            }
        }
    }
    accum
}

/// A Perlin noise generator.
pub struct Perlin {
    /// Random gradient vectors used for noise generation.
    randvec: [Vec3; Self::POINT_COUNT],
    /// Permutation table for x coordinates to shuffle the gradient vectors.
    perm_x: [usize; Self::POINT_COUNT],
    /// Permutation table for y coordinates to shuffle the gradient vectors.
    perm_y: [usize; Self::POINT_COUNT],
    /// Permutation table for z coordinates to shuffle the gradient vectors.
    perm_z: [usize; Self::POINT_COUNT],
}

/// Default implementation for Perlin noise generator, which initializes it with random gradient vectors and permutation tables.
impl Default for Perlin {
    fn default() -> Self {
        Self::new()
    }
}

impl Perlin {
    /// The number of random gradient vectors and the size of the permutation tables.
    const POINT_COUNT: usize = 256;

    pub fn new() -> Self {
        let mut rng = rand::rng();
        // Uniform sphere sampling via cylindrical projection.
        // Sample y ∈ [-1, 1] uniformly, then place the point on the sphere's
        // cross-sectional circle at that height: r = √(1 - y²), θ ∈ [0, 2π).
        // This avoids rejection sampling — constant time, no wasted iterations.
        let randvec = std::array::from_fn(|_| {
            let y = rng.random_range(-1.0..=1.0);
            let theta = rng.random_range(0.0..=std::f32::consts::TAU);
            let r = (1.0_f32 - y * y).sqrt();
            Vec3::new(r * theta.cos(), r * theta.sin(), y)
        });
        Self {
            randvec,
            perm_x: Self::generate_perm(),
            perm_y: Self::generate_perm(),
            perm_z: Self::generate_perm(),
        }
    }

    /// Computes the Perlin noise value at a given point `p` in 3D space.
    pub fn noise(&self, p: &Point3) -> f32 {
        // Computes the floor of the components of the point `p` for interpolation.
        let i = p.x().floor() as i32;
        let j = p.y().floor() as i32;
        let k = p.z().floor() as i32;

        // Computes the fractional part of the components of the point `p` for interpolation.
        let u = p.x() - i as f32;
        let v = p.y() - j as f32;
        let w = p.z() - k as f32;

        // Creates a 2x2x2 grid of gradient vectors for interpolation.
        let mut c = [[[Vec3::ZERO; 2]; 2]; 2];

        // Fills the 2x2x2 grid of gradient vectors `c` using the permutation tables and the random
        // gradient vectors.
        c.iter_mut().enumerate().for_each(|(di, x)| {
            x.iter_mut().enumerate().for_each(|(dj, y)| {
                y.iter_mut().enumerate().for_each(|(dk, z)| {
                    let ii = ((i + di as i32) & 255) as usize;
                    let jj = ((j + dj as i32) & 255) as usize;
                    let kk = ((k + dk as i32) & 255) as usize;

                    *z = self.randvec[self.perm_x[ii] ^ self.perm_y[jj] ^ self.perm_z[kk]];
                });
            });
        });

        // Performs Perlin interpolation on the grid of gradient vectors `c` at the fractional
        // coordinates `(u, v, w)`.
        perlin_interp(c, u, v, w)
    }

    /// Computes the turbulence value at a given point `point` in 3D space, with a specified depth
    /// of recursion.
    pub fn turbulence(&self, point: Point3, depth: i32) -> f32 {
        let mut tmp_point = point;
        let mut weight = 1.;
        let mut accum = 0.0;

        // Performs a recursive accumulation of noise values at scaled versions of the input point,
        // weighted by decreasing factors.
        for _ in 0..depth {
            accum += weight * self.noise(&tmp_point);
            weight *= 0.5;
            tmp_point *= 2.;
        }

        accum
    }

    /// Generates a random permutation of integers from 0 to POINT_COUNT - 1.
    fn generate_perm() -> [usize; Self::POINT_COUNT] {
        let mut p = std::array::from_fn(|i| i);
        Self::permute(&mut p);
        p
    }

    /// Randomly permutes the elements of the input array `p` in place using the Fisher-Yates
    /// shuffle
    fn permute(p: &mut [usize; Self::POINT_COUNT]) {
        let mut rng = rand::rng();
        (1..Self::POINT_COUNT).rev().for_each(|i| {
            let target = rng.random_range(0..=i);
            p.swap(i, target);
        });
    }
}
