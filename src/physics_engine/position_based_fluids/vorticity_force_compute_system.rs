use std::sync::Arc;
use ash::vk;
use crate::compute::{ComputeSystemBuilder, ComputePass};
use crate::physics_engine::PhysicsConfig;
use crate::utils::data_structures::octree::octree::Octree;
use crate::world::world_objects::{particles::Particles};
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{compute_to_graphics_memory_barrier, CommandBuffer, ShaderModule};

pub struct VorticityForceComputeSystem {
    vorticity_force_compute_pass: ComputePass,
    #[allow(unused)]
    vorticity_force_compute_shader: ShaderModule,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VorticityForceComputePushConstants {
    num_elements: u32,
    cell_size: f32,
    delta_time: f32,
    spiky_constant: f32,
    vorticity_epsilon: f32,
}

impl VorticityForceComputeSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let (vorticity_force_compute_pass, vorticity_force_compute_shader) = ComputeSystemBuilder::new(vk_core.clone(), "vorticity_force_compute")
            .entry_points(&["main"])
            .push_constants::<VorticityForceComputePushConstants>()
            // Read positions 
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Velocities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Vorticity
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Densities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Grid texture
            .add_buffer_binding(vk::DescriptorType::SAMPLED_IMAGE)
            .build_with_single_pass()?;
        Ok(
            Self{ vorticity_force_compute_pass, vorticity_force_compute_shader }
        )
    }

    pub fn execute(&mut self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, particles: &Particles, octree: &Octree, physics_config: &PhysicsConfig) {
        let particle_data = particles.buffers();

        let num_elements = particle_data.morton_codes_buffer.len() as u32;

        let push_constants = VorticityForceComputePushConstants {
            num_elements,
            cell_size: physics_config.search_radius,
            delta_time: physics_config.time_step,
            spiky_constant: physics_config.kernel_spiky_grad,
            vorticity_epsilon: physics_config.vorticity_epsilon
        };

        let device = vk_core.device();

        // Describe the buffers we want to bind
        let positions = particle_data.positions_buffer.current().vk_buffer();
        let velocities = particle_data.velocities.current().vk_buffer();
        let vorticity = particle_data.vorticity.vk_buffer();
        let densities = particle_data.densities.vk_buffer();
        
        let buffers = [positions, velocities, vorticity, densities];
        let images = [
            
        ];


        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];
        self.vorticity_force_compute_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
            &images,
            bytemuck::bytes_of(&push_constants)
        );

        let buffer_barriers = [
            compute_to_graphics_memory_barrier(
                particle_data.positions_buffer.current().vk_buffer(),
                vk_core.compute_queue_family_index(),
                vk_core.graphics_queue_family_index(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        ];

 

        let image_barrier = [];
        command_buffer.pipeline_memory_barrier2(device, &buffer_barriers, &image_barrier);
    }
}