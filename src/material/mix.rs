//! Stochastic mix of two materials.
//!
//! At each bounce, one child is chosen: `b` with probability `weight`,
//! `a` with probability `1 - weight`. The chosen child's scattering PDF
//! is used for direction generation; `eval()` and `pdf()` blend both
//! children weighted by their selection probabilities.

use std::sync::Arc;

use crate::hittable::SurfaceInteraction;
use crate::vec3::{Color3, Vec3};

use crate::material::{
    Bsdf, BsdfScatter, GPU_NONE, GpuMaterialBuffer, GpuMaterialNode, GpuMaterialType,
    MAX_BSDF_STRATS, PdfKind,
};

use super::gpu::GpuSerializable;

/// Stochastic mix of two materials. `weight` is the probability of choosing `b`.
#[derive(Clone)]
pub struct MixMaterial {
    /// Material chosen with probability `(1 - weight)`.
    pub a: Arc<dyn Bsdf>,
    /// Material chosen with probability `weight`.
    pub b: Arc<dyn Bsdf>,
    /// Selection probability for `b`. ∈ [0, 1].
    pub weight: f64,
}

impl Bsdf for MixMaterial {
    fn scatter(
        &self,
        wo: Vec3,
        si: &SurfaceInteraction,
        next_dim: &mut dyn FnMut() -> f64,
    ) -> Option<BsdfScatter> {
        let a_delta = self.a.is_delta();
        let b_delta = self.b.is_delta();

        if a_delta != b_delta {
            // Exactly one delta child → path splitting
            let delta = if a_delta { &self.a } else { &self.b };
            let d_weight = if a_delta {
                1.0 - self.weight
            } else {
                self.weight
            };
            let non_delta = if a_delta { &self.b } else { &self.a };

            let delta_result = delta.scatter(wo, si, next_dim)?;
            let BsdfScatter::Delta { wi, f_cos, eta } = delta_result else {
                unreachable!("delta child always returns Delta (is_delta() guard)")
            };
            let pk = non_delta.pdf_kind(wo, si);

            return Some(BsdfScatter::Split {
                delta_wi: wi,
                delta_f_cos: f_cos * d_weight,
                delta_eta: eta,
                non_delta_pdf_kinds: {
                    let mut arr = [None; MAX_BSDF_STRATS];
                    arr[0] = pk;
                    arr
                },
            });
        }

        let sel = next_dim();
        let (chosen, selection_prob) = if sel < self.weight {
            (self.b.as_ref() as &dyn Bsdf, self.weight)
        } else {
            (self.a.as_ref() as &dyn Bsdf, 1.0 - self.weight)
        };

        // Pass a fresh `next_dim` wrapper to the child so it can consume as many
        // dimensions as it needs (replaces the old fixed-field SampleDims).
        let mut child_next_dim = || -> f64 { next_dim() };
        let mut result = chosen.scatter(wo, si, &mut child_next_dim)?;
        // The child was selected with probability `selection_prob`. For Delta
        // paths the direction comes directly from the child (no MIS mixture),
        // so f_cos must be divided by the selection probability. NonDelta paths
        // are sampled from the integrator's mixture PDF, which doesn't depend
        // on the Mix's internal selection — eval() handles the blend.
        match &mut result {
            BsdfScatter::Delta { f_cos, .. } => {
                *f_cos /= selection_prob;
            }
            BsdfScatter::NonDelta { pdf_kinds } => {
                let other = if sel < self.weight {
                    self.a.as_ref()
                } else {
                    self.b.as_ref()
                };
                if let Some(other_kind) = other.pdf_kind(wo, si) {
                    for slot in pdf_kinds.iter_mut() {
                        if slot.is_none() {
                            *slot = Some(other_kind);
                            break;
                        }
                    }
                }
            }
            // Nested Split: the child was itself a one-delta Mix.
            // Same amplification reasoning as Delta above — stochastically
            // chosen child, Split's delta_f_cos hasn't been weighted yet.
            BsdfScatter::Split {
                delta_f_cos,
                non_delta_pdf_kinds,
                ..
            } => {
                *delta_f_cos /= selection_prob;
                let other = if sel < self.weight {
                    self.a.as_ref()
                } else {
                    self.b.as_ref()
                };
                if let Some(other_kind) = other.pdf_kind(wo, si) {
                    for slot in non_delta_pdf_kinds.iter_mut() {
                        if slot.is_none() {
                            *slot = Some(other_kind);
                            break;
                        }
                    }
                }
            }
        }
        Some(result)
    }

    fn eval(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> Color3 {
        let w = self.weight;
        // Delta children have zero eval (handled by their own eval() guard),
        // so only accumulate non-delta contributions.
        let eval_a = if self.a.is_delta() {
            Color3::ZERO
        } else {
            self.a.eval(wo, wi, si)
        };
        let eval_b = if self.b.is_delta() {
            Color3::ZERO
        } else {
            self.b.eval(wo, wi, si)
        };
        (1.0 - w) * eval_a + w * eval_b
    }

    fn pdf(&self, wo: Vec3, wi: Vec3, si: &SurfaceInteraction) -> f64 {
        let w = self.weight;
        // Delta children have zero pdf (handled by their own pdf() guard),
        // so only accumulate non-delta contributions.
        let pdf_a = if self.a.is_delta() {
            0.0
        } else {
            self.a.pdf(wo, wi, si)
        };
        let pdf_b = if self.b.is_delta() {
            0.0
        } else {
            self.b.pdf(wo, wi, si)
        };
        // Weighted mixture of the two PDFs, scaled by their selection probabilities.
        (1.0 - w) * pdf_a + w * pdf_b
    }

    fn pdf_kind(&self, wo: Vec3, si: &SurfaceInteraction) -> Option<PdfKind> {
        // Try the higher-weighted child's PDF kind first. If it has
        // no PDF kind (e.g. a delta material), fall back to the
        // other child rather than returning None.
        if self.weight > 0.5 {
            self.b.pdf_kind(wo, si).or_else(|| self.a.pdf_kind(wo, si))
        } else {
            self.a.pdf_kind(wo, si).or_else(|| self.b.pdf_kind(wo, si))
        }
    }

    fn emitted(&self, wo: Vec3, si: &SurfaceInteraction) -> Color3 {
        if !self.a.is_emissive() && !self.b.is_emissive() {
            return Color3::ZERO;
        }
        let w = self.weight;
        (1.0 - w) * self.a.emitted(wo, si) + w * self.b.emitted(wo, si)
    }

    fn is_emissive(&self) -> bool {
        self.a.is_emissive() || self.b.is_emissive()
    }

    fn reflectance_estimate(&self, wo: Vec3, si: &SurfaceInteraction) -> f64 {
        let w = self.weight;
        // Delta children have negligible albedo at non-mirror directions,
        // so only accumulate non-delta contributions.
        let r_a = if self.a.is_delta() {
            0.0
        } else {
            self.a.reflectance_estimate(wo, si)
        };
        let r_b = if self.b.is_delta() {
            0.0
        } else {
            self.b.reflectance_estimate(wo, si)
        };
        // Weighted average of the two materials' reflectance estimates.
        (1.0 - w) * r_a + w * r_b
    }

    fn is_delta(&self) -> bool {
        self.a.is_delta() && self.b.is_delta()
    }
}

impl GpuSerializable for MixMaterial {
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let a_index = self.a.serialize_gpu(buf);
        let b_index = self.b.serialize_gpu(buf);
        let params = vec![self.weight];
        let param_offset = buf.params.len() as u32;
        buf.push_params(&params);
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Mix as u32,
            param_offset,
            child_a: a_index,
            child_b: b_index,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}
