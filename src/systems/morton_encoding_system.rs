use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use crate::compute::{ComputeSystemBuilder, ComputePass};
use crate::resources::ParticleData;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};
use crate::vk_utils::compute_buffer_barrier;

/// The system transforms positions into Morton codes (u32) https://en.wikipedia.org/wiki/Z-order_curve 
pub struct MortonEncodingSystem {
    morton_encoding_pass: ComputePass,
    morton_encoding_shader: ShaderModule,
}


#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
pub struct MortonEncodingPushConstants {
    cell_size: f32
}

impl MortonEncodingSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> VkResult<Self> {
        let (morton_encoding_pass, morton_encoding_shader) = ComputeSystemBuilder::new(vk_core.clone(), "morton_encoding")
            .entry_points(&["main"])
            .push_constants::<MortonEncodingPushConstants>()
            // Positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Morton codes
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Element indexes
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            .build_with_single_pass().unwrap();
        
        Ok(
            Self {
                morton_encoding_shader,
                morton_encoding_pass
            }
        )
    }
    
    
    pub fn execute(
        &self, 
        vk_core: &Arc<VkCore>, 
        num_elements: u32, 
        cell_size: f32, 
        buffers: &ParticleData,
        command_buffer: &CommandBuffer, 
    ) {
        let push_constants = MortonEncodingPushConstants {cell_size};
        let positions = buffers.positions_buffer.current().vk_buffer();
        let morton_codes = buffers.morton_codes_buffer.vk_buffer();
        let object_indices = buffers.object_indices_buffer.vk_buffer();
        
        let buffers = [
            positions, 
            morton_codes, 
            object_indices
        ];        

        let thread_group_counts: [u32; 3] = [(num_elements + 63) / 64, 1, 1];
        
        self.morton_encoding_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
            &[],
            bytemuck::bytes_of(&push_constants)
        );
        
        barrier(vk_core, command_buffer, morton_codes, object_indices);
    }
}

fn barrier(vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, morton_codes: vk::Buffer, object_indices: vk::Buffer){
    
    let buffer_memory_barriers = [
        compute_buffer_barrier(
            morton_codes,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ
        ),
        compute_buffer_barrier(
            object_indices,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ
        )
    ];

    let dependency_info = vk::DependencyInfo::default()
        .buffer_memory_barriers(&buffer_memory_barriers);

    cmd_buffer.pipeline_barrier2(vk_core.device(), &dependency_info);

}