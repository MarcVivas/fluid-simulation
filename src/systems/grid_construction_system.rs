use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use crate::compute::ComputePass;
use crate::resources::{Particles, SpatialGrid};
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule, compute_buffer_barrier, transfer_to_compute_barrier};

pub struct GridConstructionSystem {
    grid_construction_pass: ComputePass,
    grid_construction_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct GridConstructionPushConstants{
    num_elements: u32,
    map_capacity: u32,
}

const EMPTY_KEY: u32 = 0xFFFFFFFF;

impl GridConstructionSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let grid_construction_shader = ShaderModule::new(vk_core.clone(), "grid_construction");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(grid_construction_shader.vk_shader_module())
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE);

        let bindings = [
            // Cell starts
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Cell ends
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
            // Dst Previous positions
            DescriptorSetLayoutBinding::default()
                .binding(3)
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
            .size(size_of::<GridConstructionPushConstants>() as u32)];


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

        let grid_construction_pass = ComputePass::new(
            vk_core.clone(),
            pipeline,
            pipeline_layout,
        );

        Ok(Self { grid_construction_pass, grid_construction_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, spatial_grid: &SpatialGrid, particles: &Particles) {
        let device = vk_core.device();
        let num_elements = particles.len() as u32;

        let spatial_grid_buffers = spatial_grid.buffers();
        let cell_starts = spatial_grid_buffers.cell_starts.vk_buffer();
        let cell_ends = spatial_grid_buffers.cell_ends.vk_buffer();

        let particle_buffers = particles.buffers();
        let morton_codes = particle_buffers.morton_codes_buffer.vk_buffer();

        let push_constants = GridConstructionPushConstants{
            num_elements,
            map_capacity: spatial_grid_buffers.map_capacity() as u32
        };

        self.grid_construction_pass.bind(device, command_buffer.vk_cmd_buffer());

        // First clear the hash maps before building them
        Self::clear_hash_maps(device, command_buffer, cell_starts, cell_ends);

        // Set the push constants
        command_buffer.push_constants(
            device,
            self.grid_construction_pass.pipeline_layout().vk_pipeline_layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants)
        );

        // Push the descriptors
        let descriptor_buffer_infos = [
            vk::DescriptorBufferInfo::default().buffer(cell_starts).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(cell_ends).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(morton_codes).range(vk::WHOLE_SIZE),
        ];

        let descriptor_writes = [
            // Assuming bindings [0-3) are contiguous in the set layout
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer_infos[0..descriptor_buffer_infos.len()]),
        ];

        unsafe {
            vk_core.push_descriptor().cmd_push_descriptor_set(
                command_buffer.vk_cmd_buffer(), vk::PipelineBindPoint::COMPUTE,
                self.grid_construction_pass.pipeline_layout().vk_pipeline_layout(),
                0,
                &descriptor_writes
            );
        }

        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];
        self.grid_construction_pass.dispatch(device, command_buffer.vk_cmd_buffer(), thread_group_counts);
        
        let buffer_barriers = [
            compute_buffer_barrier(
                cell_starts,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                cell_ends,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        ];
        
        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_barriers);
        
        command_buffer.pipeline_barrier2(device, &dependency_info);
    }

    fn clear_hash_maps(device: &ash::Device, command_buffer: &CommandBuffer, cell_starts: vk::Buffer, cell_ends: vk::Buffer){
        command_buffer.fill_buffer(
            device,
            cell_starts,
            0,
            vk::WHOLE_SIZE,
            EMPTY_KEY
        );
        command_buffer.fill_buffer(
            device,
            cell_ends,
            0,
            vk::WHOLE_SIZE,
            EMPTY_KEY
        );

        let src_access = vk::AccessFlags2::TRANSFER_WRITE;
        let dst_access = vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ;

        let buffer_barriers = [
            transfer_to_compute_barrier(
                cell_starts,
                src_access,
                dst_access,
            ),
            transfer_to_compute_barrier(
                cell_ends,
                src_access,
                dst_access,
            )
        ];

        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_barriers);

        command_buffer.pipeline_barrier2(device, &dependency_info);
    }
}
