# Learnings Changelog

Changelog for documents under `docs/learnings/`.

## 2026-08-04

- **sdfs-and-ray-marching.md** — Updated implementation-status sections to match the landed SDF system:
  - "Connections to raytrace-rs" now reflects the real module layout (`src/shape/sdf/`), CSG operator overloading, `TransformObject`-based transforms, and `SdfRepeat`.
  - "Pending implementation design" replaced with "Implementation status": landed features (dual-number forward-AD gradients, mean curvature, SOR over-relaxation, interior traversal, `DynEval` gradient-preserving Custom dispatch) and future direction (struct-per-primitive refactor, packet/SIMD eval after wavefront rendering, voxel adapter).
  - Normal-estimator references corrected: forward-AD dual numbers (1 eval) instead of central differences (6 evals).
