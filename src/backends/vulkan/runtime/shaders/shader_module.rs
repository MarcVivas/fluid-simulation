use crate::backends::vulkan::runtime::core::VulkanContext;

use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;

#[allow(unused_assignments, unused_labels)]
pub struct ShaderModule {
    vk_core: Arc<VulkanContext>,
    shader_module: vk::ShaderModule,
}

impl ShaderModule {
    pub fn new(vk_core: Arc<VulkanContext>, shader: super::ShaderCode) -> Result<Self> {
        let words = ash::util::read_spv(&mut std::io::Cursor::new(shader.spirv))
            .with_context(|| format!("Invalid SPIR-V for '{}'", shader.name))?;
        Self::from_words(vk_core, &words)
            .with_context(|| format!("Failed to create shader '{}'", shader.name))
    }

    fn from_words(vk_core: Arc<VulkanContext>, words: &[u32]) -> Result<Self> {
        let shader_create_info = ash::vk::ShaderModuleCreateInfo::default().code(words);
        let shader_module = unsafe {
            vk_core
                .device()
                .create_shader_module(&shader_create_info, None)
        }
        .context("failed to create shader module")?;

        Ok(Self {
            shader_module,
            vk_core,
        })
    }

    pub fn vk_shader_module(&self) -> vk::ShaderModule {
        self.shader_module
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.vk_core
                .device()
                .destroy_shader_module(self.shader_module, None);
        }
    }
}
