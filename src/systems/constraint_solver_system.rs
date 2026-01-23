use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use glam::{UVec3, UVec4};
use crate::compute::ComputePass;
use crate::resources::{Particles, SpatialGrid};
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule, compute_buffer_barrier, transfer_to_compute_barrier};

pub struct ConstraintSolverSystem {
    constraint_solver_pass: ComputePass,
    constraint_solver_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct ConstraintSolverPushConstants {
    num_elements: u32,
    cell_size: f32,
}

impl ConstraintSolverSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let constraint_solver_shader = ShaderModule::new(vk_core.clone(), "constraint_solver");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(constraint_solver_shader.vk_shader_module())
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE);

        let bindings = [
            // Read positions 
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Write positions
            DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Morton codes
            DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Grid texture
            DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
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
            .size(size_of::<ConstraintSolverPushConstants>() as u32)];


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

        let constraint_solver_pass = ComputePass::new(
            vk_core.clone(),
            pipeline,
            pipeline_layout,
        );

        Ok(Self { constraint_solver_pass, constraint_solver_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, spatial_grid: &SpatialGrid, particles: &Particles) {
        let device = vk_core.device();
        let num_elements = particles.len() as u32;

        let spatial_grid_buffers = spatial_grid.buffers();

        let particle_buffers = particles.buffers();
        let morton_codes = particle_buffers.morton_codes_buffer.vk_buffer();
        let (read_positions, write_positions) = particle_buffers.positions_buffer.read_write();
        let (read_positions, write_positions) = (read_positions.vk_buffer(), write_positions.vk_buffer());
        
        
        let push_constants = ConstraintSolverPushConstants {
            num_elements,
            cell_size: spatial_grid.cell_size(),
        };

        self.constraint_solver_pass.bind(device, command_buffer.vk_cmd_buffer());
        

        // Set the push constants
        command_buffer.push_constants(
            device,
            self.constraint_solver_pass.pipeline_layout().vk_pipeline_layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants)
        );

        // Push the descriptors
        let descriptor_buffer_infos = [
            vk::DescriptorBufferInfo::default().buffer(read_positions).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(write_positions).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(morton_codes).range(vk::WHOLE_SIZE),
        ];
        
        let descriptor_image_infos = [
            vk::DescriptorImageInfo::default()
                .image_view(spatial_grid_buffers.grid_texture_view.vk_image_view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        ];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer_infos[0..descriptor_buffer_infos.len()]),
            vk::WriteDescriptorSet::default()
                .dst_binding(descriptor_buffer_infos.len() as u32)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&descriptor_image_infos),
        ];

        unsafe {
            vk_core.push_descriptor().cmd_push_descriptor_set(
                command_buffer.vk_cmd_buffer(), vk::PipelineBindPoint::COMPUTE,
                self.constraint_solver_pass.pipeline_layout().vk_pipeline_layout(),
                0,
                &descriptor_writes
            );
        }

        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];
        self.constraint_solver_pass.dispatch(device, command_buffer.vk_cmd_buffer(), thread_group_counts);

        let buffer_barriers = [
            compute_buffer_barrier(
                read_positions,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                write_positions,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        ];

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        // Barrier from SHADER_READ_ONLY_OPTIMAL to TRANSFER_DST_OPTIMAL layout
        let image_barrier = [
            vk::ImageMemoryBarrier2::default()
                .image(spatial_grid.buffers().grid_texture.vk_image())
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .subresource_range(range)
        ];

        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_barriers)
            .image_memory_barriers(&image_barrier);

        command_buffer.pipeline_barrier2(device, &dependency_info);
    }
    
}
