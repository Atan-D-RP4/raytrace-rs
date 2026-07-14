//! GPU serialization of the material tree.
//!
//! The CPU material tree is a recursive enum. The GPU sees a flat array of
//! [`GpuMaterialNode`]s; composition variants (Mix, Coated) reference
//! children by index. The shader's switch on `material_type` mirrors the
//! CPU's match.
//!
//! Texture variation is CPU-only: the GPU buffer falls back to the material's
//! solid albedo. If you want to render a textured material on the GPU, you
//! must plumb a texture index through and have the shader sample an
//! accompanying texture buffer.

pub trait GpuSerializable {
    /// Serialize the material into the GPU buffer, returning the index of the
    /// node just pushed.
    fn serialize_gpu(&self, buf: &mut GpuMaterialBuffer) -> u32 {
        let param_offset = buf.params.len() as u32;
        buf.nodes.push(GpuMaterialNode {
            material_type: GpuMaterialType::Passthrough as u32,
            param_offset,
            child_a: GPU_NONE,
            child_b: GPU_NONE,
            texture_index: GPU_NONE,
        });
        buf.nodes.len() as u32 - 1
    }
}

/// Discriminant tag for a material in the GPU buffer. Mirrors the
/// non-composition variants of [`Material`]. Composition variants are encoded
/// via the node tree structure (`child_a` / `child_b` pointers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuMaterialType {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    DiffuseLight = 3,
    Isotropic = 4,
    Glossy = 5,
    Mix = 6,
    Coated = 7,
    /// Marks a node that exists only for its children (no parameters).
    Passthrough = 0xFFFF,
    /// Absence of material — surface contributes nothing.
    Void = 0xFFFE,
}

/// A material node in the GPU buffer.
///
/// Composition variants (Mix, Coated) reference children by index. The
/// shader walks the tree the same way the CPU's enum match does.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuMaterialNode {
    pub material_type: u32,
    pub param_offset: u32,
    pub child_a: u32,
    pub child_b: u32,
    pub texture_index: u32,
}

pub(super) const GPU_NONE: u32 = u32::MAX;

/// GPU buffer with material nodes and packed parameters.
///
/// Flattened representation of a material tree suitable for upload to GPU
/// buffers. Shader code reads nodes from `nodes` and parameters from
/// `params`.
#[derive(Debug, Default)]
pub struct GpuMaterialBuffer {
    pub nodes: Vec<GpuMaterialNode>,
    pub params: Vec<u8>,
}

impl GpuMaterialBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total byte size of the node buffer.
    pub fn node_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<GpuMaterialNode>()
    }

    pub fn push_params(&mut self, params: &[f32]) {
        // GPU uses f32, but the CPU representation keeps f32 precision. Cast here
        // so the rest of the code doesn't have to think about it.
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
