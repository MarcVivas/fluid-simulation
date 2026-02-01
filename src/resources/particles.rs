use std::error::Error;
use std::sync::Arc;
use ash::vk;
use ash::vk::{DescriptorSet};
use glam::{Vec3};
use rand::Rng;
use crate::components::{Density, FluidLambda, MortonCode, Position, Velocity, Vorticity};
use crate::renderer::Drawable;
use crate::systems::ParticleDrawingSystem;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer};
use crate::vk_utils::VkBuffer;
use crate::utils::PingPong;
use crate::vk_utils::{create_ping_pong_buffer};

pub struct Particles {
    particle_system_drawer: ParticleDrawingSystem,
    total_particles: usize,
    max_radius: f32,
    buffers: ParticleData,
    
}

pub struct ParticleData {
    pub positions_buffer: PingPong<VkBuffer<Position>>,
    pub previous_positions_buffer: PingPong<VkBuffer<Position>>,
    pub velocities: PingPong<VkBuffer<Velocity>>,
    pub lambdas: VkBuffer<FluidLambda>,
    pub densities: VkBuffer<Density>,
    pub vorticity: VkBuffer<Vorticity>,
    pub morton_codes_buffer: VkBuffer<MortonCode>,
    pub object_indices_buffer: VkBuffer<u32>,
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
        
        (0..num_particles/2).for_each(|_| {
            
            // Generate a random position
            let x_pos = random_number_generator.random_range(0.0..world_dim.x);
            let y_pos = random_number_generator.random_range(0.0..world_dim.y/2.0);
            let z_pos = random_number_generator.random_range(0.0..100.0);
            let radius = random_number_generator.random_range(3..=3) as f32;
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

        (num_particles/2..num_particles).for_each(|_| {

            // Generate a random position
            let x_pos = random_number_generator.random_range(0.0..world_dim.x);
            let y_pos = random_number_generator.random_range(0.0..world_dim.y/2.0);
            let z_pos = random_number_generator.random_range(200.0..300.0);
            let radius = random_number_generator.random_range(3..=3) as f32;
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
    
    pub fn positions_buffer(&self) -> &VkBuffer<Position> {
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
) -> Result<ParticleData, Box<dyn Error>> {
    let len = positions.len();
    
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
    
    let density_constraints = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle density constraints buffer",
    )?;
    
    let densities = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle densities",
    )?;

    let vorticity = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particles vorticity",
    )?;

    let morton_codes_buffer = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle morton codes buffer",
    )?;

    let object_indices_buffer = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle object indices buffer",
    )?;
    
    let particle_system_buffers = ParticleData {
        positions_buffer,
        previous_positions_buffer,
        velocities,
        lambdas: density_constraints,
        densities,
        vorticity,
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
