use std::sync::Arc;
use anyhow::{Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use crate::rendering::camera::CameraUniform;
use crate::vulkan::graphics::graphics_pipeline::GraphicsPipeline;
use crate::vulkan::shaders::ShaderModule;
use crate::vulkan::swapchain::window_render_target::WindowRenderTarget;
use crate::world::{particles::ParticleRenderData};
use crate::vulkan::core::VulkanContext;
use crate::vulkan::{commands::CommandBuffer, descriptors::PipelineLayout};

#[allow(unused)]
pub struct ParticleDrawer{
    vk_core: Arc<VulkanContext>,
    graphics_pipeline: GraphicsPipeline,
    task_shader_module: ShaderModule,
    mesh_shader_module: ShaderModule,
    fragment_shader_module: ShaderModule,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct PushConstants {
    view: Mat4,
    proj: Mat4,
    positions_buffer: vk::DeviceAddress,
    velocities_buffer: vk::DeviceAddress,
    num_particles: u32,
    _padding: [u32; 3]
}

impl ParticleDrawer {
    pub fn new(
        vk_core: Arc<VulkanContext>,
        render_target: &WindowRenderTarget
    ) -> Result<Self> {
        
        
        let task_shader_module = ShaderModule::new(vk_core.clone(), "particle_task_shader", None)?;

        let mesh_shader_module = ShaderModule::new(vk_core.clone(), "particle_mesh_shader", None)?;

        let fragment_shader_module = ShaderModule::new(vk_core.clone(), "particle_fragment_shader", None)?;

        let shader_stage_create_infos = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .module(task_shader_module.vk_shader_module())
                .name(c"main")
                .stage(vk::ShaderStageFlags::TASK_EXT),
            vk::PipelineShaderStageCreateInfo::default()
                .module(mesh_shader_module.vk_shader_module())
                .name(c"main")
                .stage(vk::ShaderStageFlags::MESH_EXT),
            vk::PipelineShaderStageCreateInfo::default()
                .module(fragment_shader_module.vk_shader_module())
                .name(c"main")
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];


        // Pipeline layout
        let push_const_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<PushConstants>() as u32);
            
        let pipeline_layout = PipelineLayout::new(
            vk_core.clone(),
            &[],
            &[push_const_range]
        ).context("Failed to create pipeline layout")?;

        let graphics_pipeline = GraphicsPipeline::new(
            vk_core.clone(),
            render_target,
            None,
            None,
            shader_stage_create_infos,
            pipeline_layout,
        )?;


        Ok(Self {
            vk_core,
            graphics_pipeline,
            task_shader_module,
            mesh_shader_module,
            fragment_shader_module,
        })
    }


    pub fn draw(&self, command_buffer: &CommandBuffer, total_particles: usize, particles: &ParticleRenderData, camera: &CameraUniform){
        if total_particles <= 0 {
            return;
        }
        let device = self.vk_core.device();
        // Bind pipeline
        command_buffer.bind_pipeline(
            device,
            vk::PipelineBindPoint::GRAPHICS,
            self.graphics_pipeline.graphics_pipeline()
        );

     

        let push = PushConstants {
            view: camera.view,
            proj: camera.projection,
            positions_buffer: particles.positions_address,
            velocities_buffer: particles.velocities_address,
            num_particles: particles.total_particles as u32,
            ..Default::default()
        };

        command_buffer.push_constants(
            device,
            self.graphics_pipeline.pipeline_layout().vk_pipeline_layout(),
            vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&push),
        );

        // Draw
        let group_count_x = (total_particles + 63) / 64;
        let Some(mesh_shader_loader) = self.vk_core.mesh_shader_loader() else {
            return;
        };
        unsafe {
            mesh_shader_loader.cmd_draw_mesh_tasks(command_buffer.vk_cmd_buffer(), group_count_x as u32, 1, 1);
        }

    }
}
