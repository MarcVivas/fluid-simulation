use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use crate::compute::{ComputePass};
use crate::resources::{ParticleData, Particles, SpatialGrid};
use crate::vk_core::VkCore;
use crate::vk_utils::{compute_buffer_barrier, CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};

pub struct DensityComputeSystem{
    density_compute_pass: ComputePass,
    density_compute_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
struct DensityComputePushConstants {
    num_elements: u32,
}

impl DensityComputeSystem{

    pub fn new(vk_core: &Arc<VkCore>) -> VkResult<Self> {

        let density_compute_shader = ShaderModule::new(vk_core.clone(), "density_compute");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(density_compute_shader.vk_shader_module())
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
            // Velocities
            DescriptorSetLayoutBinding::default()
                .binding(2)
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
            .size(size_of::<DensityComputePushConstants>() as u32)];


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

        let density_compute_pass = ComputePass::new(
            vk_core.clone(),
            pipeline,
            pipeline_layout,
        );

        Ok(
            Self {
                density_compute_pass, 
                density_compute_shader
            }
        )
    }

    pub fn execute(&mut self, vk_core: &Arc<VkCore>, particles: &Particles, spatial_grid: &SpatialGrid, command_buffer: &CommandBuffer) {
        let buffers = particles.buffers();
        let num_elements = buffers.morton_codes_buffer.len() as u32;
        
        let push_constants = DensityComputePushConstants {
            num_elements,
        };



        let device = vk_core.device();

        self.density_compute_pass.bind(device, command_buffer.vk_cmd_buffer());


        // Describe the buffers we want to bind
        let positions = buffers.positions_buffer.current().vk_buffer();
        let previous_positions = buffers.previous_positions_buffer.current().vk_buffer();
        let velocities = buffers.velocities.current().vk_buffer();

        let descriptor_buffer_infos = [
            vk::DescriptorBufferInfo::default().buffer(positions).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(previous_positions).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(velocities).range(vk::WHOLE_SIZE),
        ];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer_infos[0..descriptor_buffer_infos.len()]),
        ];

        unsafe {
            // PUSH the descriptor directly
            vk_core.push_descriptor().cmd_push_descriptor_set(
                command_buffer.vk_cmd_buffer(),
                vk::PipelineBindPoint::COMPUTE,
                self.density_compute_pass.pipeline_layout().vk_pipeline_layout(),
                0, // set index
                &descriptor_writes,
            );
        }

        // Set push constants
        command_buffer.push_constants(
            device,
            self.density_compute_pass.pipeline_layout().vk_pipeline_layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants)
        );




        let thread_group_counts = [((num_elements + 63) / 64), 1, 1];
        self.density_compute_pass.dispatch(device, command_buffer.vk_cmd_buffer(), thread_group_counts);

        let buffer_barriers = [
            compute_buffer_barrier(
                positions,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                previous_positions,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                velocities,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        ];

        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_barriers);

        command_buffer.pipeline_barrier2(device, &dependency_info);
    }

}