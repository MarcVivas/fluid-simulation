use crate::backends::vulkan::particles::physics::neighbors::PARTICLE_NEIGHBOR_TILE_SIZE;
const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/dfsph/density_factor_compute");

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

pub struct DensityFactorCompute {
    density_factor_compute_pass: ComputePass,
    #[allow(unused)]
    density_factor_compute_shader: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod, Default)]
struct DensityFactorComputePushConstants {
    positions: vk::DeviceAddress,
    densities: vk::DeviceAddress,
    factors: vk::DeviceAddress,
    particle_to_neighborhood: vk::DeviceAddress,
    neighbor_particle_indices: vk::DeviceAddress,
    num_workgroups: u32,
    num_elements: u32,
    kernel_radius: f32,
    poly6_constant: f32,
    kernel_radius_2: f32,
    spiky_constant: f32,
    factor_epsilon: f32,
    _padding: u32,
}

impl DensityFactorCompute {
    pub fn new(vk_core: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (density_compute_pass, density_compute_shader) =
            ComputeSystemBuilder::new(vk_core.clone(), SHADER)
                .entry_points(&["main"])
                .specialization(
                    SpecializationConstants::default()
                        .u32(THREAD_GROUP_SIZE)
                        .u32(PARTICLE_NEIGHBOR_TILE_SIZE as u32),
                )
                .push_constants::<DensityFactorComputePushConstants>()
                .build_with_single_pass()?;
        Ok(Self {
            density_factor_compute_pass: density_compute_pass,
            density_factor_compute_shader: density_compute_shader,
        })
    }

    pub fn execute(
        &mut self,
        vk_core: &VulkanContext,
        command_buffer: &CommandBuffer,
        neighbor_list: &NeighborList,
        particles: &ParticleStorage,
        physics_config: &PhysicsConfig,
    ) {
        let device = vk_core.device();
        let particle_data = particles.buffers();
        let num_elements = particle_data.hilbert_keys.len() as u32;

        // Dispatch exactly based on particle count
        let num_workgroups = (num_elements + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

        let positions = particle_data.positions_buffer.current().address();
        let densities = particle_data.densities.address();
        let factors = particle_data.dfsph.factors.address();
        let particle_to_neighborhood = neighbor_list.particle_to_neighborhood().address();
        
        let push_constants = DensityFactorComputePushConstants {
            num_elements,
            positions,
            densities,
            factors,
            particle_to_neighborhood,
            kernel_radius: physics_config.kernel_radius,
            poly6_constant: physics_config.kernel_poly6,
            kernel_radius_2: physics_config.kernel_radius_2,
            spiky_constant: physics_config.kernel_spiky_grad,
            factor_epsilon: physics_config.dfsph.factor_epsilon,
            neighbor_particle_indices: neighbor_list.neighbor_particle_indices().address(),
            num_workgroups,
            ..Default::default()
        };

        self.density_factor_compute_pass.dispatch_compute(
            vk_core,
            command_buffer,
            [num_workgroups, 1, 1],
            &[],
            &[],
            bytemuck::bytes_of(&push_constants),
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                particle_data.densities.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                particle_data.dfsph.factors.vk_buffer(),
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
