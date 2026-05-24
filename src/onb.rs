use crate::vec3::{Vec3, cross, dot, unit_vector};

pub struct Onb {
    /// u is bitangent
    u: Vec3,
    /// v is tangent
    v: Vec3,
    /// w is normal
    pub w: Vec3,
}

impl Onb {
    pub fn build_from_normal(normal: Vec3) -> Self {
        let normal = unit_vector(normal);
        let a = if normal.x.abs() < 0.9 {
            Vec3::from(1.0, 0.0, 0.0)
        } else {
            Vec3::from(0.0, 1.0, 0.0)
        };

        let v = cross(&normal, &a).unit_vector();
        let u = cross(&v, &normal);
        let w = normal;

        Onb { u, v, w }
    }

    /// Transforms a local-space vector (in tangent space) to world-space using the ONB basis.
    pub fn local_to_world(&self, local: Vec3) -> Vec3 {
        local.x * self.u + local.y * self.v + local.z * self.w
    }

    pub fn world_to_local(&self, world: Vec3) -> Vec3 {
        Vec3::from(
            dot(&world, &self.u),
            dot(&world, &self.v),
            dot(&world, &self.w),
        )
    }
}
