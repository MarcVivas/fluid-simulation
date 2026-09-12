use std::sync::Arc;

use glam::Vec4;

use crate::backends::vulkan::runtime::buffers::PingPong;
use crate::backends::vulkan::runtime::buffers::VkBuffer;
use crate::backends::vulkan::runtime::core::VulkanContext;

/// Vulkan-owned particle storage used by GPU simulation passes.
pub struct ParticleBuffers {
    pub positions_buffer: PingPong<VkBuffer<Vec4>>,
    pub velocities: PingPong<VkBuffer<Vec4>>,
    pub densities: VkBuffer<f32>,
    pub hilbert_keys: VkBuffer<u32>,
    pub particle_indexes: VkBuffer<u32>,
    pub dfsph: DfsphBuffers
}

impl ParticleBuffers {
    pub fn swap(&mut self) {
        self.positions_buffer.swap();
        self.velocities.swap();
    }
}

pub fn create_particle_data(
    vk_context: &Arc<VulkanContext>,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    positions: &[Vec4],
    velocities: &[Vec4],
) -> anyhow::Result<ParticleBuffers> {
    let len = positions.len();
    let positions_buffer = PingPong::new_vk_buffer(
        vk_context,
        positions,
        "Particle positions buffer",
        command_pool,
        queue,
    )?;
    let velocities = PingPong::new_vk_buffer(
        vk_context,
        velocities,
        "Particle velocities buffer",
        command_pool,
        queue,
    )?;

    let densities = VkBuffer::new_gpu_only_uninitialized(vk_context, len, "Particle densities")?;

    let hilbert_keys = VkBuffer::new_gpu_only_uninitialized(vk_context, len, "Particle hilbert keys")?;
    let particle_indexes = VkBuffer::new_gpu_only_uninitialized(vk_context, len, "Particle indexes buffer")?;

    let dfsph = DfsphBuffers {
        factors: VkBuffer::new_gpu_only_uninitialized(vk_context, len, "DFSPH factors")?,
        residuals: VkBuffer::new_gpu_only_uninitialized(vk_context, len, "DFSPH residuals")?,
        pressure_coefficients: VkBuffer::new_gpu_only_uninitialized(vk_context, len, "DFSPH pressure_coefficients")?,
    };

    Ok(ParticleBuffers {
        positions_buffer,
        velocities,
        densities,
        hilbert_keys,
        particle_indexes,
        dfsph
    })
}


pub struct DfsphBuffers {
    /// Converts a constraint error into a correction strength, based on the particle's neighborhood
    pub factors: VkBuffer<f32>,
    /// Stores the error still remaining in the active solve.
    pub residuals: VkBuffer<f32>,
    /// Stores the current iteration's correction coefficient, which neighboring particles read when updating velocities. 
    pub pressure_coefficients: VkBuffer<f32>,
}