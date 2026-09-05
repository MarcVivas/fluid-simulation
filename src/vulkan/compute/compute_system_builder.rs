use std::ffi::CString;
use std::sync::Arc;
use anyhow::{anyhow, Context, Result};
use ash::vk;
use ash::vk::PushConstantRange;
use crate::vulkan::compute::ComputePass;
use crate::vulkan::shaders::{ShaderCompileTimeConstants, ShaderModule};
use crate::vulkan::core::VulkanContext;
use crate::vulkan::descriptors::{DescriptorSetLayoutConfig, PipelineLayout};

/// A builder that helps create gpu compute systems.
pub struct ComputeSystemBuilder {
    vk_core: Arc<VulkanContext>,
    shader_name: String,
    entry_points: Vec<&'static str>,
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    push_constant_size: Option<u32>,
    compile_time_constants: Option<ShaderCompileTimeConstants>
}

pub struct ComputeSystemResources {
    pub compute_passes: Vec<ComputePass>,
    pub shader: ShaderModule,
}

impl ComputeSystemBuilder {
    pub fn new(vk_core: Arc<VulkanContext>, shader_name: impl Into<String>) -> ComputeSystemBuilder {
        Self {
            vk_core,
            shader_name: shader_name.into(),
            entry_points: vec![],
            bindings: vec![],
            push_constant_size: None,
            compile_time_constants: None
        }
    }

    pub fn entry_points(mut self, entry_points: &[&'static str]) -> Self {
        self.entry_points = entry_points.to_vec();
        self
    }

    pub fn add_buffer_binding(mut self, descriptor_type: vk::DescriptorType) -> Self {
        let binding = self.bindings.len() as u32;
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        );
        self
    }
    
    pub fn compile_time_constants(mut self, compile_time_constants: ShaderCompileTimeConstants) -> Self {
        self.compile_time_constants = Some(compile_time_constants);
        self        
    }

    pub fn push_constants<T: bytemuck::Pod>(mut self) -> Self{
        self.push_constant_size = Some(size_of::<T>() as u32);
        self
    }

    fn can_build(&self) -> bool {
        self.entry_points.len() > 0 
    } 
    
    pub fn build_with_single_pass(self) -> Result<(ComputePass, ShaderModule)> {
        let mut resources = self.build()?;
        Ok(
            (
                resources.compute_passes.pop().ok_or_else(|| anyhow!("Failed to get compute pass"))?,
                resources.shader
            )
        )
    }
    
    pub fn build_with_multiple_passes(self) -> Result<ComputeSystemResources> {
        self.build()
    }
    fn build(self) -> Result<ComputeSystemResources> {
        if self.can_build() == false {
            return Err(anyhow!("ComputeSystemBuilder::build() failed"));
        }
        
        let shader = ShaderModule::new(self.vk_core.clone(), &self.shader_name, self.compile_time_constants.as_ref())?;

        let descriptor_config = [
            DescriptorSetLayoutConfig {
                bindings: &self.bindings,
                flags: Some(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            }
        ];

        let push_constant_ranges: Vec<PushConstantRange> = if let Some(size) = self.push_constant_size {
            vec![
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(size)

            ]
        } else {
            Vec::default()
        };


        let mut compute_passes = Vec::with_capacity(self.entry_points.len());

        for entry_point in &self.entry_points{
            let pipeline_layout = PipelineLayout::new(self.vk_core.clone(), &descriptor_config, &push_constant_ranges)?;
            
            let shader_name = CString::new(*entry_point)
                .context("compute entry point contains an interior NUL byte")?;
            let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
                .module(shader.vk_shader_module())
                .name(shader_name.as_c_str())
                .stage(vk::ShaderStageFlags::COMPUTE);

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage_create_infos)
                .layout(pipeline_layout.vk_pipeline_layout());

            let pipeline = unsafe {
                self.vk_core.device().create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None
                )
            }.map_err(|(_, error)| error)
                .context("Failed to create compute pipeline")?[0];

            let compute_pass = ComputePass::new(
                self.vk_core.clone(),
                pipeline,
                pipeline_layout,
            );
            
            compute_passes.push(compute_pass);
        }
        
        Ok(
            ComputeSystemResources {
                compute_passes,
                shader,
            }    
        )
    }
}
