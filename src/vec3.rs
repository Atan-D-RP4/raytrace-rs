use rand::RngExt;

#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Position in 3D space.
///
/// TODO(type-safety): convert to `newtype Point3(Vec3)` to prevent accidental
/// swaps with Vec3/Color3. Add `From`, `Into`, and explicit constructor like
/// `Point3::new(x, y, z)` to preserve ergonomics.
pub type Point3 = Vec3;

/// RGB color value.
///
/// TODO(type-safety): convert to `newtype Color3(Vec3)` to prevent accidental
/// swaps with Vec3/Point3. Add `From`, `Into`, and explicit constructor like
/// `Color3::new(r, g, b)` to preserve ergonomics.
pub type Color3 = Vec3;

impl Vec3 {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub const fn from(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn random() -> Self {
        let mut rng = rand::rng();
        Self::from(rng.random(), rng.random(), rng.random())
    }

    pub fn random_range(min: f64, max: f64) -> Self {
        let mut rng = rand::rng();
        Self::from(
            rng.random_range(min..max),
            rng.random_range(min..max),
            rng.random_range(min..max),
        )
    }

    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    pub fn unit_vector(&self) -> Self {
        *self / self.length()
    }
}

// --- Operator Overrides ---

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl std::ops::Add<f64> for Vec3 {
    type Output = Vec3;

    fn add(self, t: f64) -> Vec3 {
        Self {
            x: self.x + t,
            y: self.y + t,
            z: self.z + t,
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, t: f64) -> Self {
        Self {
            x: self.x * t,
            y: self.y * t,
            z: self.z * t,
        }
    }
}

// Enables `t * v` in addition to `v * t`
impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl std::ops::Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, t: f64) -> Self {
        self * (1.0 / t)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        self.x *= t;
        self.y *= t;
        self.z *= t;
    }
}

impl std::ops::DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
        *self *= 1.0 / t;
    }
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of bounds: {}", i),
        }
    }
}

impl std::ops::IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of bounds: {}", i),
        }
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

// --- Free functions ---

#[inline(always)]
pub const fn dot(u: &Vec3, v: &Vec3) -> f64 {
    u.x * v.x + u.y * v.y + u.z * v.z
}

#[inline(always)]
pub const fn cross(u: &Vec3, v: &Vec3) -> Vec3 {
    Vec3 {
        x: u.y * v.z - u.z * v.y,
        y: u.z * v.x - u.x * v.z,
        z: u.x * v.y - u.y * v.x,
    }
}

#[inline(always)]
pub fn unit_vector(v: Vec3) -> Vec3 {
    let len = v.length();
    v / len
}

#[inline(always)]
pub fn random_in_unit_disk() -> Vec3 {
    let mut rng = rand::rng();
    loop {
        let point = Vec3::from(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            0.0,
        );
        if point.length_squared() < 1.0 {
            return point;
        }
    }
}

#[inline(always)]
pub fn random_unit_vector() -> Vec3 {
    let mut rng = rand::rng();
    loop {
        let point = Vec3::from(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        let len_squared = point.length_squared();
        if 1e-160 < len_squared && len_squared <= 1.0 {
            return point / len_squared.sqrt();
        }
    }
}

#[inline(always)]
pub fn random_in_unit_disk_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    loop {
        let point = Vec3::from(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            0.0,
        );
        if point.length_squared() < 1.0 {
            return point;
        }
    }
}

#[inline(always)]
pub fn random_unit_vector_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> Vec3 {
    loop {
        let point = Vec3::from(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        let len_squared = point.length_squared();
        if 1e-160 < len_squared && len_squared <= 1.0 {
            return point / len_squared.sqrt();
        }
    }
}

pub fn random_on_hemisphere<R: rand::Rng + ?Sized>(rng: &mut R, normal: Vec3) -> Vec3 {
    let on_unit_sphere = random_unit_vector_with_rng(rng);
    if dot(&on_unit_sphere, &normal) > 0. {
        on_unit_sphere
    } else {
        -on_unit_sphere
    }
}

#[inline(always)]
pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    *v - 2.0 * dot(v, n) * *n
}

#[inline(always)]
pub fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = dot(&(-*uv), n).min(1.0);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * *n;
    r_out_perp + r_out_parallel
}
