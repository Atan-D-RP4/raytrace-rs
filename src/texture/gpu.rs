//! GPU serialization of the texture tree.
//!
//! The CPU texture tree uses `Arc<dyn Texture>` dynamic dispatch. The GPU
//! sees a flat array of [`GpuTextureNode`]s; composition variants (Checker)
//! reference children by index. The shader's switch on `texture_type` mirrors
//! the CPU's enum match.
//!
//! Image textures are referenced by index into a separate GPU texture array
//! (populated at scene upload time). Procedural textures (Noise) are evaluated
//! in the shader using uploaded parameter buffers.

use crate::texture::TextureWrap;
use image::Rgba32FImage;

/// Sentinel value indicating a texture has no GPU representation.
pub const GPU_TEX_NONE: u32 = u32::MAX;

/// Discriminant tag for a texture in the GPU buffer. Mirrors the
/// concrete texture types on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuTextureType {
    /// 2D image texture sampled via hardware sampler.
    Image = 0,
    /// Alternates between two child textures using integer parity.
    Checker = 1,
    /// Procedural Perlin noise texture.
    Noise = 2,
    /// Texture with 3D mapping → UV generation → 2D mapping.
    Mapped = 3,
}

/// A texture node in the GPU buffer.
///
/// Composition variants (Checker) reference children by index. The
/// shader walks the tree the same way the CPU's enum match does.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuTextureNode {
    /// Texture type discriminant ([`GpuTextureType`]).
    pub texture_type: u32,
    /// Byte offset into the params buffer where this node's f32-packed
    /// parameters begin.
    pub param_offset: u32,
    /// For Checker: index of the even child texture node.
    /// Unused for other types — set to [`GPU_TEX_NONE`].
    pub child_a: u32,
    /// For Checker: index of the odd child texture node.
    /// Unused for other types — set to [`GPU_TEX_NONE`].
    pub child_b: u32,
    /// For Image: index into the GPU texture array.
    /// Unused for other types — set to [`GPU_TEX_NONE`].
    pub image_index: u32,
    /// For Image: index into the GPU sampler array.
    /// Unused for other types — set to [`GPU_TEX_NONE`].
    pub sampler_index: u32,
}

/// GPU representation of an image texture's mip chain and sampler state.
#[derive(Debug, Clone)]
pub struct ImagePayload {
    pub mips: Vec<Rgba32FImage>,
    /// Horizontal addressing mode — mirrors the CPU sampler convention
    /// ([`TextureWrap`]) so the GPU samples identically.
    pub wrap_u: TextureWrap,
    /// Vertical addressing mode — mirrors the CPU sampler convention
    /// ([`TextureWrap`]) so the GPU samples identically.
    pub wrap_v: TextureWrap,
}

/// GPU buffer with texture nodes and packed parameters.
///
/// Flattened representation of a texture tree suitable for upload to GPU
/// buffers. Shader code reads nodes from `nodes` and parameters from
/// `params`.
#[derive(Debug, Default)]
pub struct GpuTextureBuffer {
    /// Flat array of texture nodes. Indices reference children within this
    /// array (for composition) or external texture/sampler arrays (for images).
    pub nodes: Vec<GpuTextureNode>,
    /// Packed f32 parameters for all nodes. Each node's `param_offset`
    /// points into this buffer.
    pub params: Vec<u8>,
    pub images: Vec<ImagePayload>,
}

impl GpuTextureBuffer {
    /// Creates an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total byte size of the node buffer.
    pub fn node_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<GpuTextureNode>()
    }

    /// Push f32 parameters as f32-packed bytes into the params buffer.
    ///
    /// GPU uses f32, but the CPU representation keeps f32 precision. Cast
    /// here so the rest of the code doesn't have to think about it.
    ///
    /// # Safety
    ///
    /// Uses `slice::from_raw_parts` to reinterpret `[f32]` as `[u8]`.
    /// This is safe because `f32` is `#[repr(transparent)]` over `u32`
    /// and has no padding bytes — the bit pattern is valid as raw bytes.
    pub fn push_params(&mut self, params: &[f32]) {
        let floats: Vec<f32> = params.to_vec();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                floats.as_ptr() as *const u8,
                std::mem::size_of_val(&*floats),
            )
        };
        self.params.extend_from_slice(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_tex_none_sentinel() {
        assert_eq!(GPU_TEX_NONE, u32::MAX);
    }

    #[test]
    fn gpu_texture_node_layout() {
        // Verify repr(C) layout matches expected GPU alignment.
        assert_eq!(std::mem::size_of::<GpuTextureNode>(), 24); // 6 × u32
    }

    #[test]
    fn gpu_texture_buffer_empty() {
        let buf = GpuTextureBuffer::new();
        assert!(buf.nodes.is_empty());
        assert!(buf.params.is_empty());
        assert_eq!(buf.node_bytes(), 0);
    }

    #[test]
    fn gpu_texture_buffer_push_params() {
        let mut buf = GpuTextureBuffer::new();
        buf.push_params(&[0.5, 1.0, -0.25]);
        // 3 f32s = 12 bytes
        assert_eq!(buf.params.len(), 12);
    }
}
