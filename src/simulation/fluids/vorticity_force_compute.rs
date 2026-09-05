use std::sync::Arc;
use ash::vk;
use crate::simulation::neighbor_list::NeighborList;
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::simulation::physics_config::PhysicsConfig;
use crate::vulkan::shaders::ShaderModule;
use crate::world::{particles::Particles};
use crate::vulkan::core::VulkanContext;
use crate::vulkan::commands::{CommandBuffer, compute_buffer_barrier, compute_to_graphics_memory_barrier};

pub struct VorticityForceCompute {
    vorticity_force_compute_pass: ComputePass,
    #[allow(unused)]
    vorticity_force_compute_shader: ShaderModule,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct VorticityForceComputePushConstants {
    positions: vk::DeviceAddress,
    velocities: vk::DeviceAddress,
    vorticities: vk::DeviceAddress,
    densities: vk::DeviceAddress,
    particle_to_neighborhood: vk::DeviceAddress, 
    neighbor_particle_indices: vk::DeviceAddress,
    num_elements: u32,
    kernel_radius: f32,
    kernel_radius_2: f32, 
    delta_time: f32,
    spiky_constant: f32,
    vorticity_epsilon: f32,
}

impl VorticityForceCompute {
    pub fn new(vk_core: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (vorticity_force_compute_pass, vorticity_force_compute_shader) = ComputeSystemBuilder::new(vk_core.clone(), "vorticity_force_compute")
            .entry_points(&["main"])
            .push_constants::<VorticityForceComputePushConstants>()
            .build_with_single_pass()?;
        Ok(
            Self{ vorticity_force_compute_pass, vorticity_force_compute_shader }
        )
    }

    pub fn execute(&mut self, vk_core: &VulkanContext, command_buffer: &CommandBuffer, particles: &Particles, neighbor_list: &NeighborList, physics_config: &PhysicsConfig) {
        let particle_data = particles.buffers();

        let num_elements = particle_data.hilbert_keys.len() as u32;

        let push_constants = VorticityForceComputePushConstants {
            particle_to_neighborhood: neighbor_list.particle_to_neighborhood().address(),
            num_elements,
            kernel_radius: physics_config.search_radius,
            kernel_radius_2: physics_config.kernel_radius_2,
            delta_time: physics_config.time_step,
            spiky_constant: physics_config.kernel_spiky_grad,
            vorticity_epsilon: physics_config.vorticity_epsilon,
            positions: particle_data.positions_buffer.current().address(),
            velocities: particle_data.velocities.current().address(),
            vorticities: particle_data.vorticity.address(),
            densities: particle_data.densities.address(),
            neighbor_particle_indices: neighbor_list.neighbor_particle_indices().address(),
            ..Default::default()
        };

        let device = vk_core.device();
      
        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];
        self.vorticity_force_compute_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &[],
            &[],
            bytemuck::bytes_of(&push_constants)
        );

        let buffer_barriers = [
            compute_to_graphics_memory_barrier(
                particle_data.positions_buffer.current().vk_buffer(),
                vk_core.compute_queue_family_index(),
                vk_core.graphics_queue_family_index(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                particle_data.velocities.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        ];

 

        let image_barrier = [];
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &image_barrier);
    }
}
