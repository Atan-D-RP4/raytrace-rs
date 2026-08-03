use crate::math::vec3::Direction3;
use glam::{Mat3, Vec3};

/// An orthonormal basis (ONB) represented by three mutually perpendicular unit vectors: u, v, and w.
///
/// The ONB is constructed such that w is aligned with a given normal vector, and u and v are
/// tangent and bitangent vectors, respectively.
#[derive(Clone)]
pub struct Onb {
    /// u is bitangent
    u: Direction3,
    /// v is tangent
    v: Direction3,
    /// w is normal
    pub w: Direction3,
}

impl Onb {
    /// Constructs an orthonormal basis (ONB) from a given normal vector.
    ///
    /// Uses the classic method of constructing an ONB from a normal vector. The resulting basis
    /// vectors are guaranteed to be orthogonal and normalized.
    pub fn build_from_normal_legacy(normal: Direction3) -> Self {
        debug_assert!(
            normal.length_squared() >= 1e-8,
            "ONB from zero normal produces NaN basis"
        );
        let normal = normal.normalize();
        // Choose a vector that is not parallel to the normal vector
        let a = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };

        // Compute the tangent and bitangent vectors using the cross product
        let v = normal.cross(a).normalize();
        let u = v.cross(normal.into_inner());
        let w = normal;

        Onb { u, v, w }
    }

    /// Constructs an orthonormal basis (ONB) from a given normal vector.
    ///
    /// Uses the branchless or Pixar method for constructing the ONB, which is more efficient and numerically stable.
    /// Reference: Duff et al., "Building an Orthonormal Basis, Revisited," JCGT 2017
    pub fn build_from_normal(normal: Direction3) -> Self {
        debug_assert!(
            normal.length_squared() >= 1e-8,
            "ONB from zero normal produces NaN basis"
        );
        let normal = normal.normalize();
        let sign = normal.z().signum();
        let a = -1.0 / (sign + normal.z());
        let b = -normal.x() * normal.y() * a;

        let u = Direction3::new(
            1.0 + sign * normal.x() * normal.x() * a,
            sign * b,
            -sign * normal.x(),
        );
        let v = Direction3::new(b, sign + normal.y() * normal.y() * a, -normal.y());
        let w = normal;

        Onb { u, v, w }
    }

    /// Transforms a local-space vector (in tangent space) to world-space using the ONB basis.
    pub fn local_to_world(&self, local: Direction3) -> Direction3 {
        (Mat3::from_cols(
            self.u.into_inner(),
            self.v.into_inner(),
            self.w.into_inner(),
        ) * local.into_inner())
        .into()
    }

    /// Transforms a world-space vector to local-space (in tangent space) using the ONB basis.
    pub fn world_to_local(&self, world: Direction3) -> Direction3 {
        Direction3::new(
            world.dot(self.u.into_inner()),
            world.dot(self.v.into_inner()),
            world.dot(self.w.into_inner()),
        )
    }
}
