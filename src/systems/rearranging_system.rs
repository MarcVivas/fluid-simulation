use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use crate::compute::{ComputeCommandPool, ComputePass};
use crate::resources::ParticleData;
use crate::systems::IntegrationPushConstants;
use crate::vk_core::VkCore;
use crate::vk_utils::{DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};

pub struct RearrangingSystem {
    rearranging_pass: ComputePass,
    rearranging_shader: ShaderModule,
}

impl RearrangingSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let rearranging_shader = ShaderModule::new(vk_core.clone(), "integration");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(rearranging_shader.vk_shader_module())
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE);

        let bindings = [
            // Positions
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Previous positions
            DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let descriptor_set_layout_config = [DescriptorSetLayoutConfig{
            bindings: &bindings,
            flags: Some(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        }];

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<IntegrationPushConstants>() as u32)];


        let pipeline_layout = PipelineLayout::new(
            vk_core.clone(),
            &descriptor_set_layout_config,
            &push_constant_ranges
        )?;



        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage_create_infos)
            .layout(pipeline_layout.vk_pipeline_layout());

        let pipeline = unsafe {
            vk_core.device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None
            ).expect("Failed to create compute pipeline")[0]
        };

        let rearranging_pass = ComputePass::new(
            vk_core.clone(),
            pipeline,
            pipeline_layout,
        );
        
        Ok(Self { rearranging_pass, rearranging_shader })
    }
    
    pub fn execute(&self, vk_core: &Arc<VkCore>, wait_semaphores: &[vk::Semaphore], buffers: &ParticleData){
        
    }
}
