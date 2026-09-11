//! World rendering and rendering passes implemented with Vulkan.
mod config;
pub mod main_pass;
pub mod world_renderer;
pub use world_renderer::VulkanWorldRenderer;
