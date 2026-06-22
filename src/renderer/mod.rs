pub mod cpu;

pub use cpu::CpuRenderer;

use std::sync::Arc;

use crate::camera::Camera;
use crate::film::{Film, SharedFramebuffer};
use crate::hittable::{Intersectable, Sampleable};

pub trait Renderer<W, C, F>: Send + Sync
where
    W: Intersectable,
    C: Camera,
    F: Film,
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

    fn resize(&mut self, _width: u32, _height: u32) {}

    fn reset(&mut self) {}
}
