use std::sync::Arc;

use crate::camera::Camera;
use crate::film::{Film, SharedFramebuffer};
use crate::intersect::Intersectable;
use crate::light::Sampleable;

pub mod cpu;

pub use cpu::CpuRenderer;

/// A trait for rendering a scene with a given camera and film.
///
/// The renderer is responsible for generating pixel data from the scene geometry and materials,
/// using the camera's projection and the film's output format. It can optionally support
/// progressive rendering by publishing intermediate frames to a shared framebuffer.
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

    /// Resizes the renderer state for new output dimensions.
    fn resize(&mut self, _width: u32, _height: u32) {}

    /// Resets renderer state for a new scene or camera.
    fn reset(&mut self) {}
}
