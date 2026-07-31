//! Coated (layered) material: thin dielectric coating over a substrate.
//!
//! Light hits the coating first: Fresnel determines reflection vs refraction.
//! Transmitted light passes through the coating (Beer's law absorption), hits
//! the substrate, and may bounce internally between the two interfaces. Each
//! internal bounce uses a QMC-stratified Fresnel split to decide reflect/transmit.
//!
//! Key parameters: `coating_ior` (Fresnel), `coating_tint` (absorption color),
//! `thickness` (path length through the coating).

use std::f32::consts::PI;
use std::sync::Arc;

use glam::Vec3;

use crate::hittable::SurfaceInteraction;
use crate::material::gpu::GpuSerializable;
use crate::material::{
    Bsdf, BsdfScatter, GPU_NONE, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType,
    MAX_BSDF_STRATS, Material, PdfKind, fresnel_r0, fresnel_schlick, ggx_sample_h,
};
use crate::onb::Onb;
use crate::pdf::cosine_hemisphere_direction;
use crate::vec3::{Color3, Direction3};

/// Maximum number of internal bounces for a Coated material. Each bounce
/// gets a fresh dimension from `next_dim()` for the internal Fresnel split.
/// The integrator terminates the path if this limit is exceeded to avoid
/// infinite recursion.
const MAX_INTERNAL_BOUNCES: usize = 5;

/// Safety net: clamp BSDF × cosine to prevent fireflies from extreme grazing angles.
/// This is NOT a physically derived limit — it's a heuristic backstop.
/// Currently set to f32::MAX, but can be reduced if fireflies are observed in practice.
const COATED_FIRE_FLY_LIMIT: f32 = f32::MAX - 1.;

fn beers_absorption(throughput: Color3, tint: Color3, path_len: f32) -> Color3 {
    throughput * tint.powf(path_len.abs())
}

#[derive(Clone)]
pub struct CoatedMaterial {
    /// Bottom layer (absorbs transmitted light).
    pub substrate: Arc<dyn Bsdf>,
    /// Top layer (thin dielectric, reflects some light via Fresnel).
    pub coating: Arc<dyn Bsdf>,
    /// Refractive index of the coating layer (used for Fresnel).
    pub coating_ior: f32,
    /// Tint color of the coating layer (used for Beer's law absorption).
    pub coating_tint: Color3,
    /// Thickness of the coating layer (used for absorption in the coating).
    pub thickness: f32,
}

/// Result of Snell's law refraction from air into the coating layer.
struct InternalFrame {
    /// Direction inside the coating that refracts to the external direction at the top interface.
    wo_internal: Direction3,
    /// Cosine of the internal angle (dot(wo_internal, sn)).
    cos_wi_inside: f32,
}

/// Result of an internal bounce in the coating layer.
enum ScatterInternalResult {
    /// Exited the coating. Return this Delta scatter to the caller.
    Exited {
        /// Direction of the ray exiting the coating layer (toward air).
        wi: Direction3,
        /// BSDF × cosine for the exit direction, including substrate contribution.
        f_cos: Color3,
        /// Optional eta for the exit direction (None if no eta change).
        eta: Option<f32>,
    },
    /// Reflected internally. Updated wi and throughput for the next bounce.
    InternalReflection {
        /// Direction of the ray reflected internally (toward substrate).
        wi: Direction3,
        /// Updated throughput after Beer's law absorption for the internal bounce.
        throughput: Color3,
    },
}

/// Parameters for GGX substrate scatter
struct GgxScatterParams<'a> {
    /// Outgoing direction in the global frame (away from surface).
    wo_global: Direction3,
    /// Tint color of the coating layer (used for Beer's law absorption).
    coating_tint: Color3,
    /// Shading normal of the surface (points away from surface).
    shading_normal: Direction3,
    /// GGX half-vector normal from the substrate's PDF (points away from surface).
    normal: Direction3,
    /// GGX alpha parameter from the substrate's PDF.
    alpha: f32,
    /// Current throughput of the path (accumulated color).
    throughput: Color3,
    /// Current internal dimension for the Fresnel split at the coating-air boundary.
    internal_dim: f32,
    /// Surface interaction for the current bounce (used for substrate evaluation).
    si: &'a SurfaceInteraction<'a>,
    /// Function to generate the next random dimension for QMC sampling.
    next_dim: &'a mut dyn FnMut() -> f32,
}

/// Parameters for the delta substrate internal bounce helper.
struct DeltaSubstrateParams {
    /// Tint color of the coating layer (used for Beer's law absorption).
    coating_tint: Color3,
    /// Direction of the ray inside the coating layer (toward substrate).
    wi_internal: Direction3,
    /// BSDF × cosine for the internal direction from the substrate.
    f_cos_internal: Color3,
    /// Shading normal of the surface (points away from surface).
    shading_normal: Direction3,
    /// Current throughput of the path (accumulated color).
    throughput: Color3,
    /// Current internal dimension for the Fresnel split at the coating-air boundary.
    internal_dim: f32,
}

impl CoatedMaterial {
    /// Compute the internal frame direction from an external outgoing direction.
    ///
    /// Given an external `wo` (outward, away from surface), computes the direction
    /// inside the coating layer that, when refracted at the coating→air boundary,
    /// produces `wo`. Uses Snell's law: sin(θ_c) = sin(θ_a) / coating_ior.
    ///
    /// Returns `None` if total internal reflection occurs (sin_wi_inside >= 1.0).
    fn snell_internal_frame(&self, wo: Direction3, sn: Direction3) -> Option<InternalFrame> {
        let cos_wo_global = wo.dot(sn.into_inner()).max(0.0);
        let wo_perp = wo - cos_wo_global * sn;
        let sin_wo = wo_perp.into_inner().length();
        let sin_wi_inside = sin_wo / self.coating_ior;
        if sin_wi_inside >= 1.0 {
            return None; // TIR
        }
        let cos_wi_inside = (1.0 - sin_wi_inside * sin_wi_inside).max(0.0).sqrt();
        let wo_internal = if sin_wo > 1e-10 {
            let wo_unit_perp = wo_perp / sin_wo;
            cos_wi_inside * sn + sin_wi_inside * wo_unit_perp
        } else {
            sn // normal incidence
        };
        Some(InternalFrame {
            wo_internal,
            cos_wi_inside,
        })
    }

    /// Fresnel transmittance at the coating-air boundary for a given direction.
    fn fresnel_transmittance(&self, cos_theta: f32) -> f32 {
        let r0 = fresnel_r0(self.coating_ior);
        1.0 - fresnel_schlick(cos_theta, r0)
    }

    /// Beer's law absorption through the coating layer for a given internal direction.
    ///
    /// `internal_cos` is the cosine of the direction **inside** the coating (not the external/global
    /// direction). The ray travels through the coating at the internal angle — using the external
    /// cosine would give the wrong path length and incorrect absorption.
    fn coating_absorption(&self, internal_cos: f32) -> Color3 {
        let path_len = self.thickness / internal_cos.abs().max(1e-10);
        beers_absorption(Color3::ONE, self.coating_tint, path_len)
    }

    /// Handle a Delta substrate scatter result in the internal bounce loop.
    /// Applies Beer's law absorption, Fresnel split at the coating-air boundary,
    /// and either reflects internally or transmits out of the coating.
    fn scatter_delta_substrate(&self, params: DeltaSubstrateParams) -> ScatterInternalResult {
        let DeltaSubstrateParams {
            coating_tint,
            wi_internal,
            f_cos_internal,
            shading_normal,
            throughput,
            internal_dim,
        } = params;
        // Beer's law for the upward crossing through the coating layer
        let path_len_up = self.thickness / wi_internal.dot(shading_normal.into_inner()).abs();
        let throughput = beers_absorption(throughput, coating_tint, path_len_up);

        // Fresnel split at top interface (coating-air boundary),
        // using internal_dim from the caller's next_dim().
        let cos_wi_internal = wi_internal.dot(shading_normal.into_inner()).abs();
        let sin2_theta = (1.0 - cos_wi_internal * cos_wi_internal).max(0.0);
        let tir = self.coating_ior * self.coating_ior * sin2_theta > 1.0;
        let f_top_internal = fresnel_schlick(cos_wi_internal, fresnel_r0(self.coating_ior));

        if tir || internal_dim < f_top_internal {
            // Must reflect (TIR) or stochastic Fresnel reflection.
            let wi = wi_internal.reflect(shading_normal.into_inner());
            // Beer's law for the downward crossing (back through the coating)
            let path_len_down = self.thickness / wi.dot(shading_normal.into_inner()).abs();
            let throughput = beers_absorption(throughput, coating_tint, path_len_down);
            ScatterInternalResult::InternalReflection { wi, throughput }
        } else {
            // Transmit out of the coating layer, i.e., exit to air
            let exit_dir = wi_internal.refract(-shading_normal.into_inner(), self.coating_ior);
            let raw = throughput * f_cos_internal;
            // Frame-independent heuristic firefly backstop:
            // `f_cos` = BSDF × cosine should be bounded for physically valid materials
            // (Lambertian max ≈ 0.32, GGX max ≈ 2-3 at extreme grazing).
            // This is NOT a physically derived limit — it's a safety net.
            let bounded_f_cos = raw.min(Vec3::splat(COATED_FIRE_FLY_LIMIT));
            ScatterInternalResult::Exited {
                wi: exit_dir,
                f_cos: bounded_f_cos,
                eta: Some(self.coating_ior),
            }
        }
    }

    /// Handle a NonDelta GGX substrate scatter result in the internal bounce loop.
    /// Uses the GGX normal/alpha from the substrate's `PdfKind::Ggx` to importance
    /// sample a direction inside the coating, then applies Beer's law absorption,
    /// Fresnel split, and either reflects internally or transmits out.
    ///
    /// Returns `None` when TIR prevents Snell's law from finding an internal frame,
    /// or when the generated direction points away from the substrate (wrong hemisphere).
    /// In both cases the caller should fall through to the NonDelta fallback path.
    fn scatter_ggx_substrate(
        &self,
        params: &mut GgxScatterParams<'_>,
    ) -> Option<ScatterInternalResult> {
        // Compute wo_internal using Snell's law helper.
        let frame = self.snell_internal_frame(params.wo_global, params.shading_normal)?;
        let wo_int = frame.wo_internal;

        // GGX importance sampling using fresh dimensions from next_dim().
        let ggx_u = (params.next_dim)();
        let ggx_v = (params.next_dim)();
        let h_local = ggx_sample_h(params.alpha, ggx_u, ggx_v);

        let onb = Onb::build_from_normal(params.normal);
        let h_world = onb.local_to_world(h_local);

        // Reflect wo_internal about the half-vector
        let wi_int = -wo_int.reflect(h_world.into_inner());

        // Check hemisphere: wi_int must point toward the substrate
        if wi_int.dot(params.shading_normal.into_inner()) <= 0.0 {
            return None;
        }

        // Beer's law for the upward crossing
        let path_len_up = self.thickness / wi_int.dot(params.shading_normal.into_inner()).abs();
        let throughput = beers_absorption(params.throughput, params.coating_tint, path_len_up);

        // Fresnel split at top interface
        let cos_wi_int = wi_int.dot(params.shading_normal.into_inner()).abs();
        let sin2_theta = (1.0 - cos_wi_int * cos_wi_int).max(0.0);
        let tir = self.coating_ior * self.coating_ior * sin2_theta > 1.0;
        let f_top_int = fresnel_schlick(cos_wi_int, fresnel_r0(self.coating_ior));

        if tir || params.internal_dim < f_top_int {
            // Internal reflection — continue bouncing
            let wi = wi_int.reflect(params.shading_normal.into_inner());
            let path_len_down = self.thickness / wi.dot(params.shading_normal.into_inner()).abs();
            let throughput = beers_absorption(throughput, params.coating_tint, path_len_down);
            Some(ScatterInternalResult::InternalReflection { wi, throughput })
        } else {
            // Transmit out of the coating layer
            let exit_dir = wi_int.refract(-params.shading_normal.into_inner(), self.coating_ior);
            // Include the substrate's BSDF value in f_cos.
            let substrate_val = self.substrate.eval(wo_int, wi_int, params.si);
            let sub_pdf = self.substrate.pdf(wo_int, wi_int, params.si);
            let substrate_f = substrate_val / sub_pdf.max(1e-10);
            let exit_fresnel = 1.0
                - fresnel_schlick(
                    exit_dir.dot(params.shading_normal.into_inner()).abs(),
                    fresnel_r0(self.coating_ior),
                );
            let raw = throughput * substrate_f * exit_fresnel;
            // Heuristic firefly backstop: `f_cos` = BSDF × cosine should be bounded
            // for physically valid materials (Lambertian max ≈ 0.32, GGX max ≈ 2-3
            // at extreme grazing). This is NOT a physically derived limit — it's a
            // safety net.
            let bounded_f_cos = raw.min(Vec3::splat(COATED_FIRE_FLY_LIMIT));
            Some(ScatterInternalResult::Exited {
                wi: exit_dir,
                f_cos: bounded_f_cos,
                eta: Some(self.coating_ior),
            })
        }
    }
}

impl CoatedMaterial {
    /// Create a coated material (thin dielectric coating over a substrate).
    pub fn new(
        substrate: Arc<dyn Bsdf>,
        coating: Arc<dyn Bsdf>,
        coating_ior: f32,
        coating_tint: Color3,
        thickness: f32,
    ) -> Self {
        Self {
            substrate,
            coating,
            coating_ior,
            coating_tint: coating_tint.clamp(Vec3::ZERO, Vec3::ONE),
            thickness,
        }
    }
}

impl From<CoatedMaterial> for Material {
    fn from(m: CoatedMaterial) -> Self {
        Material::Coated(m)
    }
}

impl Bsdf for CoatedMaterial {
    fn scatter(
        &self,
        wo: Direction3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f32,
    ) -> Option<BsdfScatter> {
        let wo_global = wo;
        let sn = si.shading_normal();
        let mut throughput = Color3::new(1.0, 1.0, 1.0);

        // coating_tint is already clamped to [0, 1] in the constructor.
        let coating_tint = self.coating_tint;

        // Fresnel split at coating-air boundary (top interface).
        let top_fresnel = next_dim();
        let cos_wo = wo.dot(sn.into_inner()).abs();
        let f_top = fresnel_schlick(cos_wo, fresnel_r0(self.coating_ior));
        if top_fresnel < f_top {
            // Reflect off the coating, i.e., exit immediately
            return Some(BsdfScatter::Delta {
                wi: -wo.reflect(sn.into_inner()),
                f_cos: throughput,
                eta: None,
            });
        }

        // Refract into the coating layer. For IOR < 1 (e.g., sphere 1
        // with coating=dielectric(0.4)), this can TIR at shallow angles.
        // Detect TIR using Snell's law: if ri² * sin²(θ) > 1.0, TIR occurs.
        let cos_in = -wo.dot(sn.into_inner());
        let sin2_in = (1.0 - cos_in * cos_in).max(0.0);
        let ri = 1.0 / self.coating_ior;
        if ri * ri * sin2_in > 1.0 {
            // TIR: no transmission into coating. Fresnel reflect instead.
            return Some(BsdfScatter::Delta {
                wi: -wo.reflect(sn.into_inner()),
                f_cos: throughput,
                eta: None,
            });
        }
        let mut wi = (-wo).refract(sn.into_inner(), ri);
        // Beer's law absorption per crossing in the coating layer
        let path_len = self.thickness / wi.dot(sn.into_inner()).abs();
        throughput = beers_absorption(throughput, coating_tint, path_len);

        // Bounce internally between coating and substrate interfaces.
        // Each bounce gets a fresh dimension from `next_dim()` for the
        // internal Fresnel split at the coating-air boundary.
        for _bounce_idx in 0..MAX_INTERNAL_BOUNCES {
            let internal_dim = next_dim();

            // Sample the substrate material (scatter upwards).
            // Pass a fresh next_dim wrapper so the substrate consumes its own
            // dimensions independently of the coated layer's internal state.
            let mut sub_next_dim = || -> f32 { next_dim() };
            // wi instead of wo because the substrate sees the incoming direction from the coating
            // layer. sub.wi points upward (away from substrate, toward coating top)
            // Negate wi: the substrate's sample() expects wo pointing OUTWARD
            // (away from surface, toward coating), but wi points inward (toward substrate).
            let sub = self.substrate.scatter(-wi, si, &mut sub_next_dim)?;

            match sub {
                BsdfScatter::Delta {
                    wi: wi_internal,
                    f_cos: f_cos_internal,
                    ..
                } => match self.scatter_delta_substrate(DeltaSubstrateParams {
                    coating_tint,
                    wi_internal,
                    f_cos_internal,
                    shading_normal: sn,
                    throughput,
                    internal_dim,
                }) {
                    ScatterInternalResult::Exited {
                        wi: exit_wi,
                        f_cos: exit_f_cos,
                        eta: exit_eta,
                    } => {
                        return Some(BsdfScatter::Delta {
                            wi: exit_wi,
                            f_cos: exit_f_cos,
                            eta: exit_eta,
                        });
                    }
                    ScatterInternalResult::InternalReflection {
                        wi: new_wi,
                        throughput: new_throughput,
                    } => {
                        wi = new_wi;
                        throughput = new_throughput;
                    }
                },
                // NonDelta: substrate returned a PDF distribution instead of a
                // specific direction. For GGX substrates (Metal, Glossy), we
                // generate the direction in the internal frame to avoid the
                // frame mismatch between global-frame GGX sampling and
                // internal-frame eval(). For non-GGX (Cosine/Lambertian),
                // we pass through as NonDelta — the frame mismatch is benign
                // because the Lambertian eval only depends on cos(θ) · dot(n, wi).
                BsdfScatter::NonDelta {
                    pdf_kinds: sub_pdf_kinds,
                } => {
                    // Check if any pdf_kind is GGX (flatten skips Nones)
                    let ggx_info = sub_pdf_kinds.iter().flatten().find_map(|pk| {
                        if let PdfKind::Ggx { normal, alpha, .. } = pk {
                            Some((*normal, *alpha))
                        } else {
                            None
                        }
                    });

                    if let Some((normal, alpha)) = ggx_info {
                        let mut params = GgxScatterParams {
                            wo_global,
                            coating_tint,
                            shading_normal: sn,
                            normal,
                            alpha,
                            throughput,
                            internal_dim,
                            si,
                            next_dim,
                        };
                        if let Some(result) = self.scatter_ggx_substrate(&mut params) {
                            match result {
                                ScatterInternalResult::Exited {
                                    wi: exit_wi,
                                    f_cos: exit_f_cos,
                                    eta: exit_eta,
                                } => {
                                    return Some(BsdfScatter::Delta {
                                        wi: exit_wi,
                                        f_cos: exit_f_cos,
                                        eta: exit_eta,
                                    });
                                }
                                ScatterInternalResult::InternalReflection {
                                    wi: new_wi,
                                    throughput: new_throughput,
                                } => {
                                    wi = new_wi;
                                    throughput = new_throughput;
                                    continue;
                                }
                            }
                        }
                    }
                    // TIR or wrong hemisphere — fall through to NonDelta fallback

                    // Fallback for non-GGX substrates (Lambertian, etc.):
                    // Also for whenever a valid GGX half-vector produces a wrong hemisphere reflection
                    // (wi_int.dot(n) > 0.0) due to numerical issues or extreme angles.
                    //
                    // When the coating is a non-delta GGX material (Metal/Glossy with
                    // roughness > 0), include its GGX distribution as an additional MIS
                    // strategy. Without this, the Cosine-only fallback lets the coating's
                    // narrow GGX eval peak leak through with a mismatched PDF, producing
                    // fireflies.
                    if let Some(alpha) = self.coating.ggx_alpha() {
                        let mut pk = [None; MAX_BSDF_STRATS];
                        pk[0] = Some(PdfKind::Cosine { normal: sn });
                        pk[1] = Some(PdfKind::Ggx {
                            wo: wo_global,
                            normal: sn,
                            alpha,
                        });
                        return Some(BsdfScatter::NonDelta { pdf_kinds: pk });
                    }

                    // Internal-frame sampling: generate the direction in the
                    // substrate's internal coordinate frame (refracted through
                    // the coating), then refract to the global frame. Returning
                    // Delta avoids the frame mismatch that the former Cosine
                    // NonDelta fallback had between global-frame PDF and
                    // internal-frame eval.
                    let sn_inner = Direction3(-sn.into_inner());
                    let internal_onb = Onb::build_from_normal(sn_inner);
                    let (u, v) = (next_dim(), next_dim());
                    let wi_local = cosine_hemisphere_direction(u, v);
                    let wi_internal = internal_onb.local_to_world(wi_local);

                    // Refract coating → air at the top interface.
                    let wi_global = wi_internal.refract(-sn_inner.into_inner(), self.coating_ior);
                    if wi_global.length_squared() > 0.0 {
                        // Global-frame PDF: internal Cosine PDF transformed by
                        // the Jacobian dω_ext/dω_int = η²×cos_θ_int / cos_θ_ext
                        //   p_global = cos_θ_int/π * cos_θ_ext / (η² × cos_θ_int)
                        //            = cos_θ_ext / (π × η²)
                        let cos_wi_ext = wi_global.dot(sn.into_inner()).abs().max(1e-10);
                        let p_global = cos_wi_ext / (PI * self.coating_ior * self.coating_ior);
                        let eval = self.eval(wo, wi_global, si);
                        let raw_f_cos = eval * cos_wi_ext / p_global;
                        let bounded_f_cos = raw_f_cos.min(Vec3::splat(COATED_FIRE_FLY_LIMIT));
                        return Some(BsdfScatter::Delta {
                            wi: wi_global,
                            f_cos: bounded_f_cos,
                            eta: Some(self.coating_ior),
                        });
                    }
                    // TIR: no valid exit direction — terminate path
                    return None;
                }

                BsdfScatter::Split {
                    delta_wi,
                    delta_f_cos,
                    ..
                } => {
                    // Handle the delta component as above
                    match self.scatter_delta_substrate(DeltaSubstrateParams {
                        coating_tint,
                        wi_internal: delta_wi,
                        f_cos_internal: delta_f_cos,
                        shading_normal: sn,
                        throughput,
                        internal_dim,
                    }) {
                        ScatterInternalResult::Exited {
                            wi: exit_wi,
                            f_cos: exit_f_cos,
                            eta: exit_eta,
                        } => {
                            return Some(BsdfScatter::Delta {
                                wi: exit_wi,
                                f_cos: exit_f_cos,
                                eta: exit_eta,
                            });
                        }
                        ScatterInternalResult::InternalReflection {
                            wi: new_wi,
                            throughput: new_throughput,
                        } => {
                            wi = new_wi;
                            throughput = new_throughput;
                            continue;
                        }
                    }
                }
            }
        }
        None
    }

    fn eval(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> Color3 {
        let sn = si.shading_normal().into_inner();
        // Both wo and wi must be in the front-facing hemisphere.
        // Back-side directions are not reachable for a reflective coating BSDF.
        if wo.dot(sn) <= 0.0 || wi.dot(sn) <= 0.0 {
            return Color3::ZERO;
        }
        // Compute the cosine of the angle between the outgoing/incoming directions and the
        // shading normal.
        let cos_wo = wo.dot(sn);
        let cos_wi = wi.dot(sn);
        // Precompute Fresnel reflectance at normal incidence for the coating layer.
        let r0 = fresnel_r0(self.coating_ior);

        // Direct coating reflection (zero for delta coating except at mirror).
        let direct_coat = self.coating.eval(wo, wi, si);

        // Fresnel transmittance at the coating-air interface for outgoing and incoming
        // directions.
        let fresnel_o = self.fresnel_transmittance(cos_wo);
        let fresnel_i = self.fresnel_transmittance(cos_wi);

        // Refract global incoming direction into the coating's internal frame.
        let wi_internal = (-wi).refract(sn, 1.0 / self.coating_ior);
        if wi_internal.length_squared() <= 0.0 {
            // TIR — only the Fresnel reflection of the coating contributes.
            let r0 = fresnel_r0(self.coating_ior);
            let fresnel_r = fresnel_schlick(cos_wo, r0);
            let r = fresnel_r * cos_wi / std::f32::consts::PI;
            return Color3::new(r, r, r);
        }

        // Compute internal frame: direction inside coating that refracts to wo_global.
        let wo_frame = match self.snell_internal_frame(wo, sn.into()) {
            Some(f) => f,
            None => return direct_coat, // TIR — substrate invisible from this angle
        };
        let wo_internal = wo_frame.wo_internal;

        // Absorption in the coating layer (Beer's law) for outgoing and incoming paths.
        // coating_tint is already clamped to [0, 1] in the constructor.
        // wo_frame.cos_wi_inside and (-wi_internal).dot(&sn).abs() are both internal cosines
        // — same semantic, just from different sources (wo_frame vs inline).
        let coating_absorption_o = self.coating_absorption(wo_frame.cos_wi_inside);
        let coating_absorption_i = self.coating_absorption((-wi_internal).dot(sn).abs());

        // Jacobian cosine: internal-frame cosine for the incoming direction.
        let cos_wi_int = (-wi_internal).dot(sn).abs().max(1e-10);

        // Transmission coefficient/components through the coating layer (Beer's law).
        let t_o = coating_absorption_o * fresnel_o;
        let t_i = coating_absorption_i * fresnel_i;

        // Substrate contribution (single bounce, attenuated by coating absorption).
        // substrate.eval() expects both wo and wi in the outward-pointing hemisphere (dot(sn) > 0).
        // - wo_internal is already outward ✓
        // - wi_internal is inward (dot(sn) < 0), so negate it ✓
        let substrate_direct = self.substrate.eval(wo_internal, -wi_internal, si);

        // Inter-reflection correction (geometric series approximation):
        // coating-substrate-coating path and subsequent bounces.
        // Uses approximated reflectances since the exact direction changes per bounce.
        let avg_cos = 0.5;
        // Fresnel reflectance at the coating-substrate interface for the internal bounce.
        let r_top_internal = fresnel_schlick(avg_cos, r0);
        // Substrate directional-hemispherical reflectance (bounded in [0, 1]),
        // estimated from the substrate's known parameters.
        let r_sub = self.substrate.reflectance_estimate(wo_internal, si);
        // Geometric series tail r + r² + r³ + … = r/(1-r) for multi-bounce
        // inter-reflection, where r = r_sub × r_top_internal.
        // Clamped to prevent divide-by-zero from approximation errors.
        let r_prod = (r_sub * r_top_internal).clamp(0.0, 0.95);
        let series = r_prod / (1.0 - r_prod).max(1e-10);
        // Refraction Jacobian: dω_int/dω_ext = cos_ext / (η² · cos_int).
        // substrate.eval() returns the internal-frame integrand (f_r · cos_int).
        // The integrator expects the external-frame integrand (f_r_ext · cos_ext).
        // The Jacobian converts between solid-angle measures:
        //   ∫ f_int · L · cos_int dω_int = ∫ f_int · L · cos_int · (dω_int/dω_ext) dω_ext
        let cos_wi_ext = cos_wi.max(1e-10);
        let jacobian_sub = cos_wi_ext / (self.coating_ior * self.coating_ior * cos_wi_int);

        // Total contribution: direct coating reflection + transmitted substrate reflection
        // (with Jacobian) + inter-reflection correction.
        let raw = direct_coat + t_o * substrate_direct * jacobian_sub * t_i * (1.0 + series);

        // Frame-independent heuristic firefly backstop:
        // `f_cos` = BSDF × cosine should be bounded for physically valid materials
        // (Lambertian max ≈ 0.32, GGX max ≈ 2-3 at extreme grazing). This is NOT a physically derived limit — it's a safety net
        raw.min(Vec3::splat(COATED_FIRE_FLY_LIMIT))
    }

    fn pdf(&self, wo: Direction3, wi: Direction3, si: &SurfaceInteraction) -> f32 {
        let sn = si.shading_normal();
        // Both wo and wi must be in the front-facing (shading) hemisphere.
        // Back-side directions are physically unreachable as BSDF exits.
        if wo.dot(sn.into_inner()) <= 0.0 || wi.dot(sn.into_inner()) <= 0.0 {
            return 0.0;
        }
        let cos_wo = wo.dot(sn.into_inner());
        let cos_wi = wi.dot(sn.into_inner());
        // Fresnel transmittance at top interface for outgoing direction
        let fresnel_t = self.fresnel_transmittance(cos_wo);

        // Refract global incoming direction into internal frame (same as eval).
        // The Bsdf convention has wi pointing away from the surface, so -wi is the
        // incident direction for Snell's law. Unary - has lower precedence than method
        // call — (-wi).refract(...) negates the input; -wi.refract(...) would negate
        // the result, producing a direction with the wrong sign for cos_int.
        let wi_internal = (-wi).refract(sn.into_inner(), 1.0 / self.coating_ior);

        // Compute internal frame: direction inside coating that refracts to wo_global.
        let wo_frame = match self.snell_internal_frame(wo, sn) {
            Some(f) => f,
            None => return 0.0, // TIR
        };
        let wo_internal = wo_frame.wo_internal;

        // Solid-angle Jacobian for the refraction at the coating-air boundary.
        // The substrate's PDF is defined in the internal solid-angle measure (dω_int).
        // The external PDF measure (dω_ext) differs by:
        //   dω_int / dω_ext = cos(θ_air) / (η² · cos(θ_coating))
        // where θ_air is the angle of wi from sn, and θ_coating is the angle of -wi_internal from
        // sn (the outward-pointing internal direction).
        let cos_ext = cos_wi.max(1e-10);
        let cos_int = (-wi_internal).dot(sn.into_inner()).max(0.0).max(1e-10);
        // dω_int/dω_ext = cos_ext / (η² · cos_int)
        let jacobian = cos_ext / (self.coating_ior * self.coating_ior * cos_int);

        fresnel_t * self.substrate.pdf(wo_internal, -wi_internal, si) * jacobian
    }

    fn pdf_kind(&self, wo: Direction3, si: &SurfaceInteraction) -> Option<PdfKind> {
        // Refract global wo into the coating's internal frame so the
        // substrate's GGX PDF uses the same coordinates as eval/pdf.
        let sn = si.shading_normal();
        let wo_frame = self.snell_internal_frame(wo, sn)?;
        self.substrate.pdf_kind(wo_frame.wo_internal, si)
    }

    fn emitted(&self, wo: Direction3, si: &SurfaceInteraction) -> Color3 {
        let sn = si.shading_normal();
        let cos_wo = wo.dot(sn.into_inner()).abs();
        // Fresnel transmittance at coating-air boundary for the exit direction
        let fresnel_t = self.fresnel_transmittance(cos_wo);
        // Compute internal frame: direction inside coating that refracts to wo_global.
        let wo_frame = match self.snell_internal_frame(wo, sn) {
            Some(f) => f,
            None => return self.coating.emitted(wo, si), // TIR
        };
        // Beer's law absorption through the coating at the INTERNAL angle coating_tint is already
        // clamped to [0, 1] in the constructor.
        let coating_absorption = self.coating_absorption(wo_frame.cos_wi_inside);
        self.coating.emitted(wo, si)
            + coating_absorption * fresnel_t * self.substrate.emitted(wo, si)
    }

    fn is_emissive(&self) -> bool {
        self.coating.is_emissive() || self.substrate.is_emissive()
    }

    fn is_delta(&self) -> bool {
        self.coating.is_delta() && self.substrate.is_delta()
    }

    fn reflectance_estimate(&self, wo: Direction3, si: &SurfaceInteraction) -> f32 {
        // Approximate the directional-hemispherical reflectance of the coated material, averaged
        // across color channels. This is a rough estimate for MIS weighting and needn't be exact.
        let r0 = fresnel_r0(self.coating_ior);
        let coating_reflectance = 1.0 - fresnel_schlick(0.5, r0); // average angle
        let substrate_reflectance = self.substrate.reflectance_estimate(wo, si);
        coating_reflectance + (1.0 - coating_reflectance) * substrate_reflectance
    }
}

impl GpuSerializable for CoatedMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let coating_index = self.coating.serialize_gpu(buf);
        let substrate_index = self.substrate.serialize_gpu(buf);

        let params = vec![
            self.coating_ior,
            self.thickness,
            self.coating_tint.x(),
            self.coating_tint.y(),
            self.coating_tint.z(),
        ];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);

        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Coated as u32,
            param_offset,
            child_a: coating_index,
            child_b: substrate_index,
            texture_index: GPU_NONE,
        });

        buf.nodes.len() as u32 - 1
    }
}

#[cfg(test)]
mod test {
    use std::f32::consts::PI;
    use std::sync::Arc;

    use crate::hittable::SurfaceInteraction;
    use crate::material::{Bsdf, CoatedMaterial, DielectricMaterial, Material, MetalMaterial};
    use crate::vec3::{Color3, Direction3};

    /// Uniform sphere direction via spherical coordinates.
    #[inline(always)]
    pub fn uniform_sphere_sample() -> Direction3 {
        let u = rand::random::<f32>();
        let v = rand::random::<f32>();
        let phi = 2.0 * PI * u;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let z = 1.0 - 2.0 * v;
        let r = (1.0 - z * z).max(0.0).sqrt();
        Direction3::new(r * cos_phi, r * sin_phi, z)
    }

    /// PDF for uniform sphere sampling.
    #[inline(always)]
    pub fn uniform_sphere_pdf(_wi: Direction3) -> f32 {
        1.0 / (4.0 * PI)
    }

    #[test]
    fn coated_pdf_substrate_integrates_to_fresnel_t() {
        // Thick, high-IOR coating over a reflective substrate. The pdf() method
        // only returns the substrate-path component (fresnel_t(wo) * p_sub(wi_int) *
        // dω_int/dω_ext). The coating reflection component fresnel_r(wo) is handled
        // through PdfKind separately, so the total BSDF sampling PDF integrates to 1.
        // This test checks that the substrate-path component integrates to fresnel_t(wo).
        let material = CoatedMaterial::new(
            Arc::new(MetalMaterial::new(Color3::new(0.9, 0.9, 0.9), 0.1)),
            Arc::new(DielectricMaterial::tinted(1.5, Color3::new(1.0, 1.0, 1.0))),
            1.5,
            Color3::new(0.8, 0.8, 0.8),
            0.01,
        );
        // Fixed near-normal incident direction.
        let wo = Direction3::new(0.0, 0.1, 1.0).normalize();
        let mut sum = 0.0;
        let n = 500_000;

        let sn = Direction3::new(0.0, 0.0, 1.0);
        let si = SurfaceInteraction::test_surface(&Material::Void, sn);

        for _ in 0..n {
            let wi = uniform_sphere_sample();
            sum += material.pdf(wo, wi, &si) / uniform_sphere_pdf(wi);
        }
        let integral = sum / n as f32;

        // Expected integral = fresnel_t(wo) = 1 - fresnel_r(wo).
        let cos_wo = wo.dot(sn.into_inner()).max(0.0);
        let r0 = ((1.0f32 - 1.5) / (1.0 + 1.5)).powi(2); // ≈ 0.04
        let fresnel_r = r0 + (1.0 - r0) * (1.0 - cos_wo).powi(5);
        let expected = 1.0 - fresnel_r; // ≈ 0.96
        assert!(
            (integral - expected).abs() < 0.1,
            "substrate-path PDF integrates to {integral}, expected ~{expected} (fresnel_t at wo)"
        );
    }
}
