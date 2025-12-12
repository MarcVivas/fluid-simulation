use std::sync::Arc;
use ash::vk;
use ash::vk::{DescriptorSetLayoutBinding};
use glam::{Vec2, Vec4};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use crate::renderer::GraphicsPipeline;
use crate::particle_system::particle_system::ParticleSystem;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::{shader_loader, CommandBuffer, PipelineLayout};
use crate::vk_utils::vk_buffer::VkBuffer;

pub struct ParticleSystemDrawer {
    vk_core: Arc<VkCore>,
    graphics_pipeline: GraphicsPipeline,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
    quad_vertex_buffer: VkBuffer,
    quad_index_buffer: VkBuffer,
}

const QUAD_VERTICES: [Vec2; 4] = [
    // Top left
    Vec2{x: -0.5, y: -0.5},
    // Top right
    Vec2{x: 0.5, y: -0.5 },
    // Bottom right
    Vec2{x: 0.5, y: 0.5},
    // Bottom left
    Vec2{x: -0.5, y: 0.5}
];

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

impl ParticleSystemDrawer {
    pub fn new(
        vk_core: Arc<VkCore>, 
        renderer: &Renderer
    ) -> Self {

        
        let vertex_shader_module = shader_loader::load(
            vk_core.device(),
            "particle_vertex_shader",
        );

        let fragment_shader_module = shader_loader::load(
            vk_core.device(),
            "particle_fragment_shader",
        );

        let shader_stage_create_infos = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .module(vertex_shader_module)
                .name(c"main")
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(fragment_shader_module)
                .name(c"main")
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];


        // Binding 0: The quad vertices
        let binding_0_quad = vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vec2>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
            
        
        // Binding 1: The particle's positions
        let binding_1_instance = vk::VertexInputBindingDescription {
            binding: 1,
            stride: size_of::<Vec4>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        };
        
        let binding_descriptions = [binding_0_quad, binding_1_instance];

        let vertex_input_attribute_descriptions = [
            // Attribute 0: Quad 
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0
            }
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&vertex_input_attribute_descriptions);

        // Pipeline layout
        let descriptor_set_layout_binding = [DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

        let pipeline_layout = PipelineLayout::new(
            vk_core.clone(),
            &descriptor_set_layout_binding,
        ).expect("Failed to create pipeline layout");

        let graphics_pipeline = GraphicsPipeline::new(
            vk_core.clone(),
            renderer,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            vertex_input_state_info,
            shader_stage_create_infos,
            pipeline_layout,
        );

        // Create the quad buffers
        let quad_vertex_buffer = Self::create_buffer_from_slice(
            &vk_core,
            &QUAD_VERTICES,
            renderer.command_pool(),
            vk::BufferUsageFlags::VERTEX_BUFFER
        );
        
        let quad_index_buffer = Self::create_buffer_from_slice(
            &vk_core,
            &QUAD_INDICES,
            renderer.command_pool(),
            vk::BufferUsageFlags::INDEX_BUFFER
        );

        Self {
            vk_core,
            quad_index_buffer,
            quad_vertex_buffer,
            graphics_pipeline,
            vertex_shader_module,
            fragment_shader_module,
        }
    }
    

    fn create_buffer_from_slice<T: Copy>(vk_core: &Arc<VkCore>, data: &[T], command_pool: vk::CommandPool, usage: vk::BufferUsageFlags) -> VkBuffer {
        let buffer_size = (data.len() * size_of::<T>()) as vk::DeviceSize;

        let index_buffer_create_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..vk::BufferCreateInfo::default()
        };

        let allocation_create_desc = AllocationCreateDesc{
            name: "Mesh buffer",
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged
        };

        let index_buffer = VkBuffer::new(
            vk_core,
            data,
            index_buffer_create_info,
            allocation_create_desc,
            command_pool
        ).expect("Failed to create index buffer");

        index_buffer
    }

    pub fn graphics_pipeline(&self) -> &GraphicsPipeline {
        &self.graphics_pipeline
    }
    
    pub fn draw(&self, cmd_buffer: &CommandBuffer, particle_system: &ParticleSystem){
        let device = self.vk_core.device();
        // Bind pipeline
        cmd_buffer.bind_pipeline(
            device,
            vk::PipelineBindPoint::GRAPHICS,
            self.graphics_pipeline.graphics_pipeline()
        );

        // Bind vertex buffers
        let buffers = [
            self.quad_vertex_buffer.vk_buffer(),
            particle_system.positions_buffer().vk_buffer()
        ];

        let offsets = [0, 0];

        cmd_buffer.bind_vertex_buffers(
            device,
            0,
            &buffers,
            &offsets
        );

        // Bind index buffer
        cmd_buffer.bind_index_buffer(
            device,
            self.quad_index_buffer.vk_buffer(),
            0,
            vk::IndexType::UINT16
        );

        // Draw indexed
        cmd_buffer.draw_indexed(
            device,
            QUAD_INDICES.len() as u32,
            particle_system.particle_count() as u32,
            0,
            0,
            0
        );
        
    }

    pub fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, descriptor_sets: &[vk::DescriptorSet]) {
        cmd_buffer.bind_descriptor_sets(
            self.vk_core.device(),
            vk::PipelineBindPoint::GRAPHICS,
            self.graphics_pipeline().pipeline_layout().vk_pipeline_layout(),
            0,
            descriptor_sets,
            &[]
        );
    }
}

impl Drop for ParticleSystemDrawer {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_shader_module(self.vertex_shader_module, None);
            device.destroy_shader_module(self.fragment_shader_module, None);
        }
    }
}


