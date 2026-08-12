//! Scalar renderer compatibility alias.
//!
//! The scalar renderer is the B=1 specialization of `WavefrontRenderer`.

pub type CpuRenderer<I> = super::wavefront::WavefrontRenderer<I, 1>;
