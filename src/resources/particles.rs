use std::error::Error;
use std::sync::Arc;
use ash::vk;
use ash::vk::{DescriptorSet};
use glam::{Vec3};
use rand::Rng;
use crate::components::{DensityConstraint, MortonCode, Position, Velocity};
use crate::renderer::Drawable;
use crate::systems::ParticleDrawingSystem;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer};
use crate::vk_utils::VkBuffer;
use crate::utils::PingPong;
use crate::vk_utils::{create_gpu_only_buffer, create_ping_pong_buffer};

pub struct Particles {
    particle_system_drawer: ParticleDrawingSystem,
    total_particles: usize,
    max_radius: f32,
    buffers: ParticleData,
    
}

pub struct ParticleData {
    pub positions_buffer: PingPong<VkBuffer>,
    pub previous_positions_buffer: PingPong<VkBuffer>,
    pub velocities: PingPong<VkBuffer>,
    pub density_constraints: VkBuffer,
    pub morton_codes_buffer: VkBuffer,
    pub object_indices_buffer: VkBuffer,
}

impl ParticleData {
    pub fn swap(&mut self) {
        self.positions_buffer.swap();
        self.previous_positions_buffer.swap();
        self.velocities.swap();
    }
}

impl Particles {
    pub fn new(
        num_particles: usize, 
        world_dim: &Vec3, 
        vk_core: &Arc<VkCore>, 
        renderer: &Renderer,
    ) -> Result<Self, Box<dyn Error>> {
        let mut random_number_generator = rand::rng(); 
        
        let mut max_radius :f32  = 0.0;

        let mut positions: Vec<Position> = Vec::with_capacity(num_particles);
        let mut previous_positions: Vec<Position> = Vec::with_capacity(num_particles);
        let mut velocities: Vec<Velocity> = Vec::with_capacity(num_particles);
        let density_constraints: Vec<f32> = vec![0.0; num_particles];
        
        (0..num_particles).for_each(|_| {
            
            // Generate a random position
            let x_pos = random_number_generator.random_range(0.0..world_dim.x);
            let y_pos = random_number_generator.random_range(0.0..world_dim.y);
            let z_pos = random_number_generator.random_range(0.0..world_dim.z);
            let radius = random_number_generator.random_range(2..4) as f32;
            max_radius = max_radius.max(radius);
            let position = Position::new(x_pos, y_pos, z_pos, radius);
            positions.push(position);
            previous_positions.push(position);
            
            // Generate a random velocity
            static MAX_VELOCITY: f32 = 5.0;
            let x = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let y = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let z = random_number_generator.random_range(0.0..MAX_VELOCITY);
            let velocity = Velocity::new(x, y, z, 0.0);
            velocities.push(velocity);
        });
        
        let morton_codes: Vec<MortonCode> = vec![0; positions.len()];
        let object_indices: Vec<u32> = vec![0; positions.len()];
        
        let particle_system_drawer = ParticleDrawingSystem::new(
            vk_core.clone(),
            renderer
        );
        
        let particle_system_buffers = create_particle_data(
            vk_core, 
            renderer.command_pool(),
            *vk_core.graphics_queue(),
            &positions, 
            &previous_positions, 
            &velocities,
            &density_constraints,
            &morton_codes, 
            &object_indices
        )?;
        
        Ok(
            Self {
                total_particles: num_particles,
                buffers: particle_system_buffers,
                particle_system_drawer,
                max_radius
            }
        )
    }
    
    pub fn positions_buffer(&self) -> &VkBuffer {
        &self.buffers.positions_buffer.current()
    }
    
    pub fn len(&self) -> usize {
        self.total_particles
    }
    
    pub fn buffers(&self) -> &ParticleData {
        &self.buffers
    }
    
    pub fn buffers_mut(&mut self) -> &mut ParticleData {
        &mut self.buffers
    }
    
    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }
}

fn create_particle_data(
    vk_core: &Arc<VkCore>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    positions: &[Position],
    previous_positions: &[Position],
    velocities: &[Velocity],
    density_constraints: &[DensityConstraint],
    morton_codes: &[MortonCode],
    object_indices: &[u32],
) -> Result<ParticleData, Box<dyn Error>> {
    let positions_buffer = create_ping_pong_buffer(
        vk_core,
        positions,
        "Particle positions buffer",
        command_pool,
        queue
    )?;

    let previous_positions_buffer =  create_ping_pong_buffer(
        vk_core,
        previous_positions,
        "Particle previous positions buffer",
        command_pool,
        queue
    )?;
    
    let velocities = create_ping_pong_buffer(
        vk_core,
        velocities,
        "Particle velocities buffer",
        command_pool,
        queue
    )?;
    
    let density_constraints = create_gpu_only_buffer(
        vk_core,
        density_constraints,
        "Particle density constraints buffer",
        command_pool,
        queue
    )?;

    let morton_codes_buffer = create_gpu_only_buffer(
        vk_core,
        morton_codes,
        "Particle morton codes buffer",
        command_pool,
        queue
    )?;

    let object_indices_buffer = create_gpu_only_buffer(
        vk_core,
        object_indices,
        "Particle object indices buffer",
        command_pool,
        queue
    )?;
    
    let particle_system_buffers = ParticleData {
        positions_buffer,
        previous_positions_buffer,
        velocities,
        density_constraints,
        morton_codes_buffer,
        object_indices_buffer,
    };
    
    Ok(particle_system_buffers)
}


impl Drawable for Particles {
    fn draw(&self, cmd_buffer: &CommandBuffer) {
        self.particle_system_drawer.draw(cmd_buffer, &self);
    }

    fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, global_descriptor_set: &[DescriptorSet]) {
        self.particle_system_drawer.bind_descriptor_sets(cmd_buffer, global_descriptor_set, &self.buffers);
    }
}
