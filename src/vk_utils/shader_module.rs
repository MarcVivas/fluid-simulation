use std::ops::Deref;
use std::sync::Arc;
use ash::vk;
use crate::vk_core::VkCore;
use crate::vk_utils::shader_loader;

pub struct ShaderModule {
    vk_core: Arc<VkCore>,
    shader_module: vk::ShaderModule
    
}

impl ShaderModule {
    pub fn new(vk_core: Arc<VkCore>, shader_name: &str) -> Self {
        let shader_module = shader_loader::load(vk_core.device(), shader_name);
        Self{shader_module, vk_core}
    }
    
    pub fn vk_shader_module(&self) -> vk::ShaderModule {
        self.shader_module
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe { self.vk_core.device().destroy_shader_module(self.shader_module, None); }
    }
    
}
