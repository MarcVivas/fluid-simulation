#[derive(Clone, Copy)]
pub enum BackendKind {
    Vulkan,
}

pub mod gpu_backend;
pub mod vulkan;
