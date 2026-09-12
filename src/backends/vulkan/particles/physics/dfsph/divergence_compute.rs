use crate::backends::vulkan::particles::physics::neighbors::PARTICLE_NEIGHBOR_TILE_SIZE;
const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/dfsph/divergence_compute");

use crate::backends::vulkan::particles::ParticleStorage;
use crate::backends::vulkan::particles::physics::neighbors::NeighborList;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::compute_buffer_barrier;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;
use crate::world::particles::physics_config::PhysicsConfig;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

pub struct DivergenceCompute {
    divergence_compute_pass: ComputePass,
    #[allow(unused)]
    divergence_compute_shader: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod, Default)]
struct DivergenceComputePushConstants {
    positions: vk::DeviceAddress,
    velocities: vk::DeviceAddress,
    factors: vk::DeviceAddress,
    residuals: vk::DeviceAddress,
    pressure_coefficients: vk::DeviceAddress,
    particle_to_neighborhood: vk::DeviceAddress,
    neighbor_particle_indices: vk::DeviceAddress,
    num_elements: u32,
    kernel_radius: f32,
    spiky_constant: f32,
    delta_time: f32,
}

impl DivergenceCompute {
    pub fn new(vk_context: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (divergence_compute_pass, divergence_compute_shader) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .specialization(
                    SpecializationConstants::default()
                        .u32(THREAD_GROUP_SIZE)
                        .u32(PARTICLE_NEIGHBOR_TILE_SIZE as u32),
                )
                .push_constants::<DivergenceComputePushConstants>()
                .build_with_single_pass()?;
        Ok(Self {
            divergence_compute_pass,
            divergence_compute_shader,
        })
    }

    pub fn execute(
        &mut self,
        vk_context: &VulkanContext,
        command_buffer: &CommandBuffer,
        neighbor_list: &NeighborList,
        particles: &ParticleStorage,
        physics_config: &PhysicsConfig,
    ) {
        let device = vk_context.device();
        let particle_data = particles.buffers();
        let num_elements = particle_data.hilbert_keys.len() as u32;

        // Dispatch exactly based on particle count
        let num_workgroups = (num_elements + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

        let positions = particle_data.positions_buffer.current().address();
        let factors = particle_data.dfsph.factors.address();
        let particle_to_neighborhood = neighbor_list.particle_to_neighborhood().address();
        
        let push_constants = DivergenceComputePushConstants {
            num_elements,
            velocities: particle_data.velocities.current().address(),
            residuals: particle_data.dfsph.residuals.address(),
            pressure_coefficients: particle_data.dfsph.pressure_coefficients.address(),
            positions,
            factors,
            particle_to_neighborhood,
            kernel_radius: physics_config.kernel_radius,
            spiky_constant: physics_config.kernel_spiky_grad,
            neighbor_particle_indices: neighbor_list.neighbor_particle_indices().address(),
            delta_time: physics_config.time_step,
            ..Default::default()
        };

        self.divergence_compute_pass.dispatch_compute(
            vk_context,
            command_buffer,
            [num_workgroups, 1, 1],
            &[],
            &[],
            bytemuck::bytes_of(&push_constants),
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                particle_data.dfsph.residuals.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                particle_data.dfsph.pressure_coefficients.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &["main"],
        2,
    );
}
