//! Vulkan implementation of engine features and their execution.
mod backend;
pub use backend::VulkanBackend;

pub mod algorithms;
pub mod particles;
pub mod rendering;
pub mod runtime;
pub mod world_resources;
