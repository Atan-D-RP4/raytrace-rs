//! Coated (layered) material: thin dielectric coating over a substrate.
//!
//! Light hits the coating first: Fresnel determines reflection vs refraction.
//! Transmitted light passes through the coating (Beer's law absorption), hits
//! the substrate, and may bounce internally between the two interfaces. Each
//! internal bounce uses a QMC-stratified Fresnel split to decide reflect/transmit.
//!
//! Key parameters: `coating_ior` (Fresnel), `coating_tint` (absorption color),
//! `thickness` (path length through the coating).

use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::material::gpu::GpuSerializable;
use crate::material::{
    Bsdf, BsdfScatter, GPU_NONE, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType, PdfKind,
    fresnel_r0, fresnel_schlick, ggx_sample_h, reflect, refract,
};
use crate::onb::Onb;
use crate::sampler::SampleDims;
use crate::vec3::{Color3, Vec3};

/// Maximum number of internal bounces for a Coated material, matching the
/// number of QMC dimensions reserved for internal Fresnel splits (dims.v
/// through dims.z, one per bounce). The integrator terminates the path if
/// this limit is exceeded to avoid infinite recursion.
const MAX_INTERNAL_BOUNCES: usize = 5;

fn beers_absorption(throughput: Color3, tint: Color3, path_len: f64) -> Color3 {
    throughput
        * Color3::new(
            tint.x.powf(path_len.abs()),
            tint.y.powf(path_len.abs()),
            tint.z.powf(path_len.abs()),
        )
}

#[derive(Clone)]
pub struct CoatedMaterial {
    /// Bottom layer (absorbs transmitted light).
    pub substrate: Arc<dyn Bsdf>,
    /// Top layer (thin dielectric, reflects some light via Fresnel).
    pub coating: Arc<dyn Bsdf>,
    /// Refractive index of the coating layer (used for Fresnel).
    pub coating_ior: f64,
    /// Tint color of the coating layer (used for Beer's law absorption).
    pub coating_tint: Color3,
    /// Thickness of the coating layer (used for absorption in the coating).
    pub thickness: f64,
}

impl Bsdf for CoatedMaterial {
    fn scatter(&self, wo: Vec3, si: &SurfaceInteraction, dims: SampleDims) -> Option<BsdfScatter> {
        let wo_global = wo;
        let n = si.shading_normal();
        let mut throughput = Color3::new(1.0, 1.0, 1.0);

        // Clamp coating tint to [0, 1] per component.
        // Values > 1 would amplify via powf (physically invalid Beer's law).
        let coating_tint = Color3::new(
            self.coating_tint.x.clamp(0.0, 1.0),
            self.coating_tint.y.clamp(0.0, 1.0),
            self.coating_tint.z.clamp(0.0, 1.0),
        );

        // Fresnel split at coating-air boundary (top interface).
        // Uses dims.u as the QMC-stratified Fresnel threshold.
        let cos_wo = wo.dot(&n).abs();
        let f_top = fresnel_schlick(cos_wo, fresnel_r0(self.coating_ior));
        if dims.u < f_top {
            // Reflect off the coating, i.e., exit immediately
            return Some(BsdfScatter::Delta {
                wi: reflect(&-wo, &n),
                f_cos: throughput,
            });
        }

        // Refract into the coating layer. For IOR < 1 (e.g., sphere 1
        // with coating=dielectric(0.4)), this can TIR at shallow angles.
        // Detect TIR using Snell's law: if ri² * sin²(θ) > 1.0, TIR occurs.
        let cos_in = -wo.dot(&n);
        let sin2_in = (1.0 - cos_in * cos_in).max(0.0);
        let ri = 1.0 / self.coating_ior;
        if ri * ri * sin2_in > 1.0 {
            // TIR: no transmission into coating. Fresnel reflect instead.
            return Some(BsdfScatter::Delta {
                wi: reflect(&-wo, &n),
                f_cos: throughput,
            });
        }
        let mut wi = refract(&-wo, &n, ri);
        // Beer's law absorption per crossing in the coating layer
        let path_len = self.thickness / wi.dot(&n).abs();
        throughput = beers_absorption(throughput, coating_tint, path_len);

        // Internal Fresnel splits use dims.v through dims.z (one per bounce).
        // The substrate gets default dims — for Delta substrates (metal,
        // dielectric) this is fine; NonDelta exits the walk immediately.
        let internal_dims = [dims.v, dims.w, dims.x, dims.y, dims.z];

        // Bounce internally between coating and substrate interfaces
        for (bounce_idx, internal_dim) in
            internal_dims.iter().enumerate().take(MAX_INTERNAL_BOUNCES)
        {
            // Sample the substrate material (scatter upwards)
            // Each bounce gets a shifted QMC dimension to preserve stratification.
            let bounce_offset = bounce_idx as f64 * 0.123456789;
            let sub_u = (dims.u + bounce_offset).fract();
            let sub_v = (dims.v + bounce_offset * 2.0).fract();
            let sub_w = (dims.w + bounce_offset * 3.0).fract();
            let sub_dims = SampleDims {
                u: sub_u,
                v: sub_v,
                w: sub_w,
                x: dims.x,
                y: dims.y,
                z: dims.z,
            };
            // wi instead of wo because the substrate sees the incoming direction from the
            // coating layer
            // sub.wi points upward (away from substrate, toward coating top)
            // Negate wi: the substrate's sample() expects wo pointing OUTWARD
            // (away from surface, toward coating), but wi points inward (toward substrate).
            let sub = self.substrate.scatter(-wi, si, sub_dims)?;

            match sub {
                BsdfScatter::Delta {
                    wi: wi_internal,
                    f_cos: f_cos_internal,
                } => {
                    // Beer's law for the upward crossing through the coating layer
                    let path_len_up = self.thickness / wi_internal.dot(&n).abs();
                    throughput = Color3::new(
                        throughput.x * coating_tint.x.powf(path_len_up.abs()),
                        throughput.y * coating_tint.y.powf(path_len_up.abs()),
                        throughput.z * coating_tint.z.powf(path_len_up.abs()),
                    );

                    // Fresnel split at top interface (coating-air boundary),
                    // using a QMC-stratified threshold from internal_dims.
                    let cos_wi_internal = wi_internal.dot(&n).abs();
                    let sin2_theta = (1.0 - cos_wi_internal * cos_wi_internal).max(0.0);
                    let tir = self.coating_ior * self.coating_ior * sin2_theta > 1.0;
                    let f_top_internal =
                        fresnel_schlick(cos_wi_internal, fresnel_r0(self.coating_ior));

                    if tir || *internal_dim < f_top_internal {
                        // Must reflect (TIR) or stochastic Fresnel reflection.
                        wi = reflect(&wi_internal, &n);
                        // Beer's law for the downward crossing (back through the coating)
                        let path_len_down = self.thickness / wi.dot(&n).abs();
                        throughput = Color3::new(
                            throughput.x * coating_tint.x.powf(path_len_down.abs()),
                            throughput.y * coating_tint.y.powf(path_len_down.abs()),
                            throughput.z * coating_tint.z.powf(path_len_down.abs()),
                        );
                    } else {
                        // Transmit out of the coating layer, i.e., exit to air
                        let exit_dir = refract(&wi_internal, &-n, self.coating_ior);
                        let raw = throughput * f_cos_internal;
                        // Frame-independent heuristic firefly backstop:
                        // `f_cos` = BSDF × cosine should be bounded
                        let bounded_f_cos =
                            Color3::new(raw.x.min(2.0), raw.y.min(2.0), raw.z.min(2.0));
                        return Some(BsdfScatter::Delta {
                            wi: exit_dir,
                            f_cos: bounded_f_cos,
                        });
                    }
                }
                // NonDelta: substrate returned a PDF distribution instead of a
                // specific direction. For GGX substrates (Metal, Glossy), we
                // generate the direction in the internal frame to avoid the
                // frame mismatch between global-frame GGX sampling and
                // internal-frame eval(). For non-GGX (Cosine/Lambertian),
                // we pass through as NonDelta — the frame mismatch is benign
                // because the Lambertian eval only depends on cos(θ) · dot(n, wi).
                BsdfScatter::NonDelta {
                    pdf_kinds: sub_pdf_kinds,
                    count: sub_count,
                } => {
                    // Check if any pdf_kind is GGX
                    let ggx_info = sub_pdf_kinds[..sub_count as usize].iter().find_map(|pk| {
                        if let PdfKind::Ggx { normal, alpha, .. } = pk {
                            Some((*normal, *alpha))
                        } else {
                            None
                        }
                    });

                    if let Some((normal, alpha)) = ggx_info {
                        // Compute wo_internal: the direction inside the coating
                        // that refracts to wo_global at the top interface.
                        let cos_wo_g = wo_global.dot(&n).max(0.0);
                        let wo_perp = wo_global - cos_wo_g * n;
                        let sin_wo = wo_perp.length();
                        let sin_w_in = sin_wo / self.coating_ior;
                        let cos_w_in = (1.0 - sin_w_in * sin_w_in).max(0.0).sqrt();
                        let wo_int = if sin_wo > 1e-10 {
                            let wo_unit_perp = wo_perp / sin_wo;
                            cos_w_in * n + sin_w_in * wo_unit_perp
                        } else {
                            n
                        };

                        // GGX importance sampling using the internal wo.
                        // Uses the same inverse-CDF as Metal/Glossy.
                        let h_local = ggx_sample_h(alpha, dims.u, dims.v);

                        let onb = Onb::build_from_normal(normal);
                        let h_world = onb.local_to_world(h_local);

                        // Reflect wo_internal about the half-vector
                        let wi_int = reflect(&-wo_int, &h_world);

                        // Check hemisphere: wi_int must point toward the substrate
                        if wi_int.dot(&n) > 0.0 {
                            // Beer's law for the upward crossing
                            let path_len_up = self.thickness / wi_int.dot(&n).abs();
                            throughput = Color3::new(
                                throughput.x * coating_tint.x.powf(path_len_up.abs()),
                                throughput.y * coating_tint.y.powf(path_len_up.abs()),
                                throughput.z * coating_tint.z.powf(path_len_up.abs()),
                            );

                            // Fresnel split at top interface
                            let cos_wi_int = wi_int.dot(&n).abs();
                            let sin2_theta = (1.0 - cos_wi_int * cos_wi_int).max(0.0);
                            let tir = self.coating_ior * self.coating_ior * sin2_theta > 1.0;
                            let f_top_int =
                                fresnel_schlick(cos_wi_int, fresnel_r0(self.coating_ior));

                            if tir || *internal_dim < f_top_int {
                                // Internal reflection — continue bouncing
                                wi = reflect(&wi_int, &n);
                                let path_len_down = self.thickness / wi.dot(&n).abs();
                                throughput = Color3::new(
                                    throughput.x * coating_tint.x.powf(path_len_down.abs()),
                                    throughput.y * coating_tint.y.powf(path_len_down.abs()),
                                    throughput.z * coating_tint.z.powf(path_len_down.abs()),
                                );
                                continue;
                            }

                            // Transmit out of the coating layer
                            let exit_dir = refract(&wi_int, &-n, self.coating_ior);
                            // Include the substrate's BSDF value in f_cos.
                            // wo_internal is the outgoing direction in the internal frame,
                            // wi_int is the substrate's outgoing direction (toward coating).
                            let substrate_val = self.substrate.eval(wo_int, wi_int, si);
                            // Use the substrate's own PDF for consistency with its eval().
                            // This avoids mismatches between a hand-rolled PDF derivation
                            // and the substrate's internal conventions.
                            let sub_pdf = self.substrate.pdf(wo_int, wi_int, si);
                            let substrate_f = substrate_val / sub_pdf.max(1e-10);
                            let exit_fresnel = 1.0
                                - fresnel_schlick(
                                    exit_dir.dot(&n).abs(),
                                    fresnel_r0(self.coating_ior),
                                );
                            // Heuristic firefly backstop: `f_cos` = BSDF × cosine should be bounded
                            // for physically valid materials, such as:
                            // (Lambertian max ≈ 0.32, GGX max ≈ 2-3 at extreme grazing).
                            // This is NOT a physically derived limit — it's a safety net.
                            let raw = throughput * substrate_f * exit_fresnel;
                            let bounded_f_cos =
                                Color3::new(raw.x.min(2.0), raw.y.min(2.0), raw.z.min(2.0));
                            return Some(BsdfScatter::Delta {
                                wi: exit_dir,
                                f_cos: bounded_f_cos,
                            });
                        }
                    }

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
                        return Some(BsdfScatter::NonDelta {
                            pdf_kinds: [
                                PdfKind::Cosine { normal: n },
                                PdfKind::Ggx {
                                    wo: wo_global,
                                    normal: n,
                                    alpha,
                                },
                            ],
                            count: 2,
                        });
                    }

                    return Some(BsdfScatter::NonDelta {
                        pdf_kinds: [PdfKind::Cosine { normal: n }, PdfKind::Delta],
                        count: 1,
                    });
                }
            }
        }
        None
    }

    fn eval(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> Color3 {
        let sn = si.shading_normal();
        // Compute the cosine of the angle between the outgoing/incoming directions and the
        // shading normal.
        let cos_wo = wo.dot(&sn).abs();
        let cos_wi = wi.dot(&sn).abs();
        // Precompute Fresnel reflectance at normal incidence for the coating layer.
        let r0 = fresnel_r0(self.coating_ior);

        // Direct coating reflection (zero for delta coating except at mirror).
        let direct_coat = self.coating.eval(wo, wi, si);

        // Fresnel reflectance at the coating-air interface for outgoing and incoming
        // directions.
        let fresnel_o = 1.0 - fresnel_schlick(cos_wo, r0);
        let fresnel_i = 1.0 - fresnel_schlick(cos_wi, r0);

        // Refract global incoming direction into the coating's internal frame.
        let wi_internal = refract(&-wi, &sn, 1.0 / self.coating_ior);

        // Compute the internal exit direction: the direction inside the coating
        // that, when refracted at the coating-air interface from coating→air (IOR = coating_ior),
        // becomes the global wo direction (outward, dot(sn) > 0).
        // From Snell's law: sin(θ_c) = sin(θ_a) / coating_ior
        let cos_wo_global = wo.dot(&sn).max(0.0);
        let wo_perp = wo - cos_wo_global * sn;
        let sin_wo = wo_perp.length();
        let sin_wi_inside = sin_wo / self.coating_ior;
        if sin_wi_inside > 1.0 {
            // TIR — no transmission to air.
            return direct_coat;
        }
        let cos_wi_inside = (1.0 - sin_wi_inside * sin_wi_inside).max(0.0).sqrt();
        let wo_internal = if sin_wo > 1e-10 {
            // Tangent direction in coating is the same as in air (just scaled)
            let wo_unit_perp = wo_perp / sin_wo;
            cos_wi_inside * sn + sin_wi_inside * wo_unit_perp
        } else {
            // Normal incidence — straight through
            sn
        };

        // Path lengths through the coating layer for outgoing and incoming directions.
        // The ray travels at the INTERNAL angle inside the coating, so use the internal
        // direction's cosine (not the global direction's cosine) for correct Beer's law.
        let cos_wi_int = (-wi_internal).dot(&sn).abs().max(1e-10);
        let cos_wo_int = wo_internal.dot(&sn).abs().max(1e-10);
        let path_o = self.thickness / cos_wo_int;
        let path_i = self.thickness / cos_wi_int;

        // Absorption in the coating layer (Beer's law) for outgoing and incoming paths.
        // Clamp tint components to [0, 1] to prevent amplification (tint > 1 would
        // add energy via powf).
        let tint = Color3::new(
            self.coating_tint.x.clamp(0.0, 1.0),
            self.coating_tint.y.clamp(0.0, 1.0),
            self.coating_tint.z.clamp(0.0, 1.0),
        );
        let coating_absorption_o = Color3::new(
            tint.x.powf(path_o),
            tint.y.powf(path_o),
            tint.z.powf(path_o),
        );
        let coating_absorption_i = Color3::new(
            tint.x.powf(path_i),
            tint.y.powf(path_i),
            tint.z.powf(path_i),
        );

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
        Color3::new(raw.x.min(2.0), raw.y.min(2.0), raw.z.min(2.0))
    }

    fn pdf(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> f64 {
        let sn = si.shading_normal();
        let cos_wo = wo.dot(&sn).abs();
        let cos_wi = wi.dot(&sn).abs();
        let r0 = fresnel_r0(self.coating_ior);

        // Fresnel transmittance at top interface for outgoing direction
        let fresnel_t = 1.0 - fresnel_schlick(cos_wo, r0);

        // Refract global incoming direction into internal frame (same as eval)
        let wi_internal = refract(&-wi, &sn, 1.0 / self.coating_ior);

        // Snell reversal for wo_internal (same as eval)
        let cos_wo_global = wo.dot(&sn).max(0.0);
        let wo_perp = wo - cos_wo_global * sn;
        let sin_wo = wo_perp.length();
        let sin_wi_inside = sin_wo / self.coating_ior;
        if sin_wi_inside >= 1.0 {
            // TIR: substrate is invisible from this angle.
            return 0.0;
        }
        let cos_wi_inside = (1.0 - sin_wi_inside * sin_wi_inside).max(0.0).sqrt();
        // Compute the internal outgoing direction that refracts to wo_global at the top interface.
        let wo_internal = if sin_wo > 1e-10 {
            // Tangent direction in coating is the same as in air (just scaled)
            let wo_unit_perp = wo_perp / sin_wo;
            cos_wi_inside * sn + sin_wi_inside * wo_unit_perp
        } else {
            sn // normal incidence
        };

        // Solid-angle Jacobian for the refraction at the coating-air boundary.
        // The substrate's PDF is defined in the internal solid-angle measure (dω_int).
        // The external PDF measure (dω_ext) differs by:
        //   dω_int / dω_ext = cos(θ_air) / (η² · cos(θ_coating))
        // where θ_air is the angle of wi from sn, and θ_coating is the angle
        // of -wi_internal from sn (the outward-pointing internal direction).
        let cos_ext = cos_wi.max(1e-10);
        let cos_int = (-wi_internal).dot(&sn).max(0.0).max(1e-10);
        // dω_int/dω_ext = cos_ext / (η² · cos_int)
        let jacobian = cos_ext / (self.coating_ior * self.coating_ior * cos_int);

        fresnel_t * self.substrate.pdf(wo_internal, -wi_internal, si) * jacobian
    }

    fn pdf_kind(&self, wo: Vec3, si: &SurfaceInteraction) -> Option<PdfKind> {
        // Refract global wo into the coating's internal frame so the
        // substrate's GGX PDF uses the same coordinates as eval/pdf.
        let sn = si.shading_normal();
        let cos_wo_global = wo.dot(&sn).max(0.0);
        let wo_perp = wo - cos_wo_global * sn;
        let sin_wo = wo_perp.length();
        let sin_wi_inside = sin_wo / self.coating_ior;
        if sin_wi_inside >= 1.0 {
            // TIR: substrate is invisible from this angle.
            return None;
        }
        let cos_wi_inside = (1.0 - sin_wi_inside * sin_wi_inside).sqrt();
        let wo_internal = if sin_wo > 1e-10 {
            let wo_unit_perp = wo_perp / sin_wo;
            cos_wi_inside * sn + sin_wi_inside * wo_unit_perp
        } else {
            sn // normal incidence
        };
        self.substrate.pdf_kind(wo_internal, si)
    }

    fn emitted(&self, wo: Vec3, si: &SurfaceInteraction) -> Color3 {
        let sn = si.shading_normal();
        let cos_wo = wo.dot(&sn).abs();
        let r0 = fresnel_r0(self.coating_ior);
        // Fresnel transmittance at coating-air boundary for the exit direction
        let fresnel_t = 1.0 - fresnel_schlick(cos_wo, r0);
        // Refract wo into the coating's internal frame to get the internal angle.
        // wo points outward. From Snell's law: sin(θ_c) = sin(θ_a) / coating_ior
        let cos_wo_global = wo.dot(&sn).max(0.0);
        let wo_perp = wo - cos_wo_global * sn;
        let sin_wo = wo_perp.length();
        let sin_wi_inside = sin_wo / self.coating_ior;
        if sin_wi_inside >= 1.0 {
            // TIR: substrate is invisible from this angle.
            return self.coating.emitted(wo, si);
        }
        let cos_wi_inside = (1.0 - sin_wi_inside * sin_wi_inside).max(0.0).sqrt();
        let wo_internal = if sin_wo > 1e-10 {
            let wo_unit_perp = wo_perp / sin_wo;
            cos_wi_inside * sn + sin_wi_inside * wo_unit_perp
        } else {
            sn
        };
        // Beer's law absorption through the coating at the INTERNAL angle
        let cos_wo_int = wo_internal.dot(&sn).abs().max(1e-10);
        let path_o = self.thickness / cos_wo_int;
        let tint = Color3::new(
            self.coating_tint.x.clamp(0.0, 1.0),
            self.coating_tint.y.clamp(0.0, 1.0),
            self.coating_tint.z.clamp(0.0, 1.0),
        );
        let coating_absorption = Color3::new(
            tint.x.powf(path_o),
            tint.y.powf(path_o),
            tint.z.powf(path_o),
        );
        self.coating.emitted(wo, si)
            + coating_absorption * fresnel_t * self.substrate.emitted(wo, si)
    }

    fn is_emissive(&self) -> bool {
        self.coating.is_emissive() || self.substrate.is_emissive()
    }

    fn is_delta(&self) -> bool {
        self.coating.is_delta() && self.substrate.is_delta()
    }

    fn reflectance_estimate(&self, _wo: Vec3, _si: &SurfaceInteraction) -> f64 {
        // Approximate the directional-hemispherical reflectance of the coated material, averaged
        // across color channels.
        // This is a rough estimate for MIS weighting and does not need to be exact.
        let r0 = fresnel_r0(self.coating_ior);
        let coating_reflectance = 1.0 - fresnel_schlick(0.5, r0); // average angle
        let substrate_reflectance = self.substrate.reflectance_estimate(_wo, _si);
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
            self.coating_tint.x,
            self.coating_tint.y,
            self.coating_tint.z,
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
