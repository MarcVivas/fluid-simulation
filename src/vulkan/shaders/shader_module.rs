use crate::vulkan::core::VkCore;
use crate::vulkan::shaders::loader::get_shader_bytecode;
use crate::vulkan::shaders::constants::ShaderCompileTimeConstants;

use ash::vk;
use std::sync::Arc;

#[allow(unused_assignments, unused_labels)]
pub struct ShaderModule {
    vk_core: Arc<VkCore>,
    shader_module: vk::ShaderModule,
}

impl ShaderModule {
    pub fn new(vk_core: Arc<VkCore>, shader_name: &str, compile_time_constants: Option<&ShaderCompileTimeConstants>) -> Self {
        let spirv_code = get_shader_bytecode(shader_name, compile_time_constants.unwrap_or(&ShaderCompileTimeConstants::default())); 
                
        let shader_create_info = ash::vk::ShaderModuleCreateInfo::default()
            .code(&spirv_code);
        let shader_module = unsafe {
            vk_core.device().create_shader_module(
                &shader_create_info,
                None,
            )
        }.expect("failed to create shader module");
        
        Self {
            shader_module,
            vk_core,
        }
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
