use crate::vulkan::core::VulkanContext;
use crate::vulkan::shaders::loader::get_shader_bytecode;
use crate::vulkan::shaders::constants::ShaderCompileTimeConstants;

use ash::vk;
use anyhow::{Context, Result};
use std::sync::Arc;

#[allow(unused_assignments, unused_labels)]
pub struct ShaderModule {
    vk_core: Arc<VulkanContext>,
    shader_module: vk::ShaderModule,
}

impl ShaderModule {
    pub fn new(vk_core: Arc<VulkanContext>, shader_name: &str, compile_time_constants: Option<&ShaderCompileTimeConstants>) -> Result<Self> {
        let default_constants = ShaderCompileTimeConstants::default();
        let constants = compile_time_constants.unwrap_or(&default_constants);
        let spirv_code = get_shader_bytecode(shader_name, constants)
            .with_context(|| format!("Failed to load shader '{shader_name}'"))?;
                
        let shader_create_info = ash::vk::ShaderModuleCreateInfo::default()
            .code(&spirv_code);
        let shader_module = unsafe {
            vk_core.device().create_shader_module(
                &shader_create_info,
                None,
            )
        }.context("failed to create shader module")?;
        
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
