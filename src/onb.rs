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
    /// Constructs an orthonormal basis (ONB) from a given normal vector.
    ///
    /// Uses the classic method of constructing an ONB from a normal vector. The resulting basis
    /// vectors are guaranteed to be orthogonal and normalized.
    pub fn build_from_normal_legacy(normal: Vec3) -> Self {
        debug_assert!(
            !normal.near_zero(),
            "ONB from zero normal produces NaN basis"
        );
        let normal = normal.unit_vector();
        // Choose a vector that is not parallel to the normal vector
        let a = if normal.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };

        // Compute the tangent and bitangent vectors using the cross product
        let v = normal.cross(&a).unit_vector();
        let u = v.cross(&normal);
        let w = normal;

        Onb { u, v, w }
    }

    /// Constructs an orthonormal basis (ONB) from a given normal vector.
    ///
    /// Uses the branchless or Pixar method for constructing the ONB, which is more efficient and numerically stable.
    /// Reference: Duff et al., "Building an Orthonormal Basis, Revisited," JCGT 2017
    pub fn build_from_normal(normal: Vec3) -> Self {
        debug_assert!(
            !normal.near_zero(),
            "ONB from zero normal produces NaN basis"
        );
        let normal = normal.unit_vector();
        let sign = normal.z.signum();
        let a = -1.0 / (sign + normal.z);
        let b = -normal.x * normal.y * a;

        let u = Vec3::new(
            1.0 + sign * normal.x * normal.x * a,
            sign * b,
            -sign * normal.x,
        );
        let v = Vec3::new(b, sign + normal.y * normal.y * a, -normal.y);
        let w = normal;

        Onb { u, v, w }
    }

    /// Transforms a local-space vector (in tangent space) to world-space using the ONB basis.
    pub fn local_to_world(&self, local: Vec3) -> Vec3 {
        local.x * self.u + local.y * self.v + local.z * self.w
    }

    /// Transforms a world-space vector to local-space (in tangent space) using the ONB basis.
    pub fn world_to_local(&self, world: Vec3) -> Vec3 {
        Vec3::new(world.dot(&self.u), world.dot(&self.v), world.dot(&self.w))
    }
}
