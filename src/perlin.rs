use rand::RngExt;

use crate::vec3::{Point3, Vec3, dot, random_unit_vector};

#[allow(dead_code)]
fn trilinear_interp(c: [[[f64; 2]; 2]; 2], u: f64, v: f64, w: f64) -> f64 {
    (0..2)
        .flat_map(|i| (0..2).flat_map(move |j| (0..2).map(move |k| (i, j, k))))
        .fold(0.0, |acc, (i, j, k)| {
            acc + (i as f64 * u + (1 - i) as f64 * (1.0 - u))
                * (j as f64 * v + (1 - j) as f64 * (1.0 - v))
                * (k as f64 * w + (1 - k) as f64 * (1.0 - w))
                * c[i][j][k]
        })
}

pub fn perlin_interp(c: [[[Vec3; 2]; 2]; 2], u: f64, v: f64, w: f64) -> f64 {
    // Hermite smoothing
    let u = u * u * (3.0 - 2.0 * u);
    let v = v * v * (3.0 - 2.0 * v);
    let w = w * w * (3.0 - 2.0 * w);

    let mut accum = 0.0;
    // Using for-loop instead of iterators for better optimization
    for i in 0..2 {
        let fu = if i == 1 { u } else { 1.0 - u };
        for j in 0..2 {
            let fv = if j == 1 { v } else { 1.0 - v };
            for k in 0..2 {
                let fw = if k == 1 { w } else { 1.0 - w };
                let weight = Vec3::from(u - i as f64, v - j as f64, w - k as f64);
                accum += fu * fv * fw * dot(&c[i][j][k], &weight);
            }
        }
    }
    accum
}

pub struct Perlin {
    randvec: [Vec3; Self::POINT_COUNT],
    perm_x: [usize; Self::POINT_COUNT],
    perm_y: [usize; Self::POINT_COUNT],
    perm_z: [usize; Self::POINT_COUNT],
}

impl Default for Perlin {
    fn default() -> Self {
        Self::new()
    }
}

impl Perlin {
    const POINT_COUNT: usize = 256;

    pub fn new() -> Self {
        Self {
            randvec: std::array::from_fn(|_| random_unit_vector()),
            perm_x: Self::generate_perm(),
            perm_y: Self::generate_perm(),
            perm_z: Self::generate_perm(),
        }
    }

    pub fn noise(&self, p: &Point3) -> f64 {
        let i = p.x.floor() as i32;
        let j = p.y.floor() as i32;
        let k = p.z.floor() as i32;

        let u = p.x - i as f64;
        let v = p.y - j as f64;
        let w = p.z - k as f64;

        let mut c = [[[Vec3::new(); 2]; 2]; 2];

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

        perlin_interp(c, u, v, w)
    }

    pub fn turbulence(&self, point: Point3, depth: i32) -> f64 {
        let mut tmp_point = point;
        let mut weight = 1.;
        let mut accum = 0.0;

        for _ in 0..depth {
            accum += weight * self.noise(&tmp_point);
            weight *= 0.5;
            tmp_point *= 2.;
        }

        accum
    }

    fn generate_perm() -> [usize; Self::POINT_COUNT] {
        let mut p = std::array::from_fn(|i| i);
        Self::permute(&mut p);
        p
    }

    fn permute(p: &mut [usize; Self::POINT_COUNT]) {
        let mut rng = rand::rng();
        (1..Self::POINT_COUNT).rev().for_each(|i| {
            let target = rng.random_range(0..=i);
            p.swap(i, target);
        });
    }
}
