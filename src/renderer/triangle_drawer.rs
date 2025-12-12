use std::mem::offset_of;
use std::sync::Arc;
use ash::vk;
use ash::vk::{DescriptorSetLayoutBinding};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use crate::renderer::drawable::Drawable;
use crate::vk_utils::{shader_loader, CommandBuffer};
use crate::renderer::graphics_pipeline::GraphicsPipeline;
use crate::renderer::renderer::Renderer;
use crate::vk_core::vk_core::VkCore;
use crate::vk_utils::pipeline_layout::PipelineLayout;
use crate::vk_utils::vk_buffer::VkBuffer;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Vertex {
    pos: glam::Vec4,
    color: glam::Vec4,
}

pub struct TriangleDrawer {
    vk_core: Arc<VkCore>,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    index_buffer: VkBuffer,
    vertex_buffer: VkBuffer,
    graphics_pipeline: GraphicsPipeline,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
}

impl TriangleDrawer {
    pub fn new(vk_core: Arc<VkCore>, renderer: &Renderer) -> Self {


        let vertices = vec![
            Vertex {
                pos: glam::Vec4::new(-0.5, 0.5, 0.0, 1.0),
                color: glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            },
            Vertex {
                pos: glam::Vec4::new(0.5, 0.5, 0.0, 1.0),
                color: glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
            },
            Vertex {
                pos: glam::Vec4::new(0.5, -0.5, 0.0, 1.0),
                color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            },
            Vertex {
                pos: glam::Vec4::new(-0.5, -0.5, 0.0, 1.0),
                color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            },
        ];
        

        let vertex_shader_module = shader_loader::load(
            vk_core.device(),
            "vertex_shader",
        );

        let fragment_shader_module = shader_loader::load(
            vk_core.device(),
            "fragment_shader",
        );

        let shader_stage_create_infos = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .module(vertex_shader_module)
                .name(c"main")
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo {
                module: fragment_shader_module,
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_name: c"main".as_ptr() as *const i8,
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..vk::PipelineShaderStageCreateInfo::default()
            }
        ];



        let vertex_input_binding_description = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }
        ];

        let vertex_input_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32
            }
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_input_binding_description)
            .vertex_attribute_descriptions(&vertex_input_attribute_descriptions);


        let descriptor_set_layout_binding = [DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        
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
        
        let vertex_buffer = TriangleDrawer::create_vertex_buffer(
            &vk_core,
            &vertices,
            renderer.command_pool()
        );

        let indices: Vec<u16>  = vec![0, 1, 2, 2, 3, 0];

        let index_buffer = TriangleDrawer::create_index_buffer(
            &vk_core,
            &indices,
            renderer.command_pool()
        );
        
        Self {
            vk_core,
            vertices,
            indices,
            index_buffer,
            vertex_buffer,
            graphics_pipeline,
            vertex_shader_module,
            fragment_shader_module,
        }
    }
    
    fn create_vertex_buffer(vk_core: &Arc<VkCore>, vertices: &Vec<Vertex>, command_pool: vk::CommandPool) -> VkBuffer {
        let buffer_size = (vertices.len() * size_of::<Vertex>()) as vk::DeviceSize;
        
        let vertex_buffer_create_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..vk::BufferCreateInfo::default()
        };

        let allocation_create_desc = AllocationCreateDesc{
            name: "Vertex Buffer",
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged
        };
        
       let vertex_buffer = VkBuffer::new(
           vk_core,
           vertices,
           vertex_buffer_create_info,
           allocation_create_desc,
           command_pool
       ).expect("Failed to create vertex buffer");

        vertex_buffer
    }
    
    fn create_index_buffer(vk_core: &Arc<VkCore>, indices: &Vec<u16>, command_pool: vk::CommandPool) -> VkBuffer {
        let buffer_size = (indices.len() * size_of::<u16>()) as vk::DeviceSize;

        let index_buffer_create_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..vk::BufferCreateInfo::default()
        };

        let allocation_create_desc = AllocationCreateDesc{
            name: "Index Buffer",
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged
        };

        let index_buffer = VkBuffer::new(
            vk_core,
            indices,
            index_buffer_create_info,
            allocation_create_desc,
            command_pool
        ).expect("Failed to create index buffer");

        index_buffer
    }
    
    pub fn graphics_pipeline(&self) -> &GraphicsPipeline {
        &self.graphics_pipeline
    }
}

impl Drawable for TriangleDrawer {
    fn draw(&self, cmd_buffer: &CommandBuffer) {
        let device = self.vk_core.device();
        cmd_buffer.bind_pipeline(device, vk::PipelineBindPoint::GRAPHICS, self.graphics_pipeline.graphics_pipeline());
        cmd_buffer.bind_vertex_buffers(device, 0, &[self.vertex_buffer.vk_buffer()], &[0]);
        cmd_buffer.bind_index_buffer(device, self.index_buffer.vk_buffer(), 0, vk::IndexType::UINT16);
        cmd_buffer.draw_indexed(device, self.indices.len() as u32, 1, 0, 0, 1);
    }

    fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, descriptor_sets: &[vk::DescriptorSet]) {
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

impl Drop for TriangleDrawer {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_shader_module(self.vertex_shader_module, None);
            device.destroy_shader_module(self.fragment_shader_module, None);
        }
    }
}
