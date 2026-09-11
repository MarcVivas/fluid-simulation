use std::sync::Arc;

use crate::backends::vulkan::particles::ParticleRenderInput;
use crate::backends::vulkan::particles::ParticleStorage;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::world::World;
use crate::world::WorldBounds;

/// Vulkan storage initialized from the logical world and shared by physics and rendering.
pub struct VulkanWorldResources {
    state: WorldBounds,
    particle_system: ParticleStorage,
}

impl VulkanWorldResources {
    pub fn new(
        vk_core: &Arc<VulkanContext>,
        world: &World,
        command_pool: ash::vk::CommandPool,
    ) -> anyhow::Result<Self> {
        let state = WorldBounds::new(&world.dimensions());

        let particle_system =
            ParticleStorage::from_state(vk_core, command_pool, world.initial_particles())?;

        Ok(Self {
            particle_system,
            state,
        })
    }

    pub fn state(&self) -> WorldBounds {
        self.state
    }

    pub fn particles(&self) -> &ParticleStorage {
        &self.particle_system
    }

    pub fn particles_mut(&mut self) -> &mut ParticleStorage {
        &mut self.particle_system
    }

    pub fn extract_render_data(&self) -> ParticleRenderInput {
        self.particle_system.extract_render_data()
    }
}
