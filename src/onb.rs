use crate::vec3::Vec3;

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
        let normal = normal.unit_vector();
        let a = if normal.x.abs() < 0.9 {
            Vec3::from(1.0, 0.0, 0.0)
        } else {
            Vec3::from(0.0, 1.0, 0.0)
        };

        let v = normal.cross(&a).unit_vector();
        let u = v.cross(&normal);
        let w = normal;

        Onb { u, v, w }
    }

    /// Transforms a local-space vector (in tangent space) to world-space using the ONB basis.
    pub fn local_to_world(&self, local: Vec3) -> Vec3 {
        local.x * self.u + local.y * self.v + local.z * self.w
    }

    pub fn world_to_local(&self, world: Vec3) -> Vec3 {
        Vec3::from(world.dot(&self.u), world.dot(&self.v), world.dot(&self.w))
    }
}
