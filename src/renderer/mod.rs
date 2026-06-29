pub mod cpu;

pub use cpu::CpuRenderer;

use std::sync::Arc;

use crate::camera::Camera;
use crate::film::{Film, SharedFramebuffer};
use crate::hittable::{Intersectable, Sampleable};
use crate::sampler::Sampler;

pub trait Renderer<W, C, F, S>: Send + Sync
where
    W: Intersectable,
    C: Camera,
    F: Film,
    S: Sampler,
{
    /// Renders the scene and returns (width, height, RGB pixel data).
    ///
    /// When `framebuffer` is `Some`, publishes progressive intermediate frames
    /// to the shared framebuffer during rendering (live preview mode).
    /// When `None`, renders all samples and returns the final image only.
    fn render(
        &self,
        camera: &C,
        film: &mut F,
        scene: (&W, &[Arc<dyn Sampleable>]),
        framebuffer: Option<SharedFramebuffer>,
    );

    /// Resizes the renderer state for new output dimensions.
    fn resize(&mut self, _width: u32, _height: u32) {}

    /// Resets renderer state for a new scene or camera.
    fn reset(&mut self) {}
}
