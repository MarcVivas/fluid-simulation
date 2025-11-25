use std::mem::offset_of;
use std::sync::Arc;
use ash::util::Align;
use ash::vk;
use ash::vk::{RenderPassBeginInfo};
use crate::{shader_loader, utils};
use crate::renderer::graphics_pipeline::GraphicsPipeline;
use crate::renderer::renderer::Renderer;
use crate::vk_core::vk_core::VkCore;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

pub struct TriangleDrawer {
    vk_core: Arc<VkCore>,
    vertices: [Vertex; 3],
    indices: [u32; 3],
    index_buffer: vk::Buffer,
    vertex_input_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    vertex_input_buffer_memory: vk::DeviceMemory,
    graphics_pipeline: GraphicsPipeline,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
}

impl TriangleDrawer {
    pub fn new(vk_core: Arc<VkCore>, renderer: &Renderer) -> Self {


        let vertices = [
            Vertex {
                pos: [-0.5, 0.5, 0.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                pos: [0.5, 0.5, 0.0, 1.0],
                color: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                pos: [0.0, -0.5, 0.0, 1.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
        ];

        let indices: [u32; 3] = [0, 1, 2];
        let index_buffer_create_info = vk::BufferCreateInfo::default()
            .size(size_of_val(&indices) as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let index_buffer = unsafe {
            vk_core.device().create_buffer(&index_buffer_create_info, None)
        }.expect("failed to create index buffer");

        let index_buffer_memory_requirements = unsafe {
            vk_core.device().get_buffer_memory_requirements(index_buffer)
        };

        let device_memory_properties = vk_core.device_memory_properties();

        let index_buffer_memory_index = utils::find_memory_type_index(
            &index_buffer_memory_requirements,
            device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).expect("failed to find suitable memory type for the index buffer");

        let index_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: index_buffer_memory_requirements.size,
            memory_type_index: index_buffer_memory_index,
            ..vk::MemoryAllocateInfo::default()
        };

        let index_buffer_memory = unsafe {
            vk_core
                .device()
                .allocate_memory(&index_allocate_info, None)
        }.expect("failed to allocate memory for the index buffer");

        let index_ptr = unsafe {
            vk_core.device().map_memory(
                index_buffer_memory,
                0,
                index_buffer_memory_requirements.size,
                vk::MemoryMapFlags::empty()
            )
        }.expect("failed to map memory");

        let mut index_slice = unsafe {
            Align::new(
                index_ptr,
                align_of::<u32>() as u64,
                index_buffer_memory_requirements.size
            )
        };
        index_slice.copy_from_slice(&indices);
        unsafe {
            vk_core.device().unmap_memory(index_buffer_memory);
            vk_core
                .device()
                .bind_buffer_memory(index_buffer, index_buffer_memory, 0)
                .expect("failed to bind index buffer memory");
        }

        let vertex_input_buffer_create_info = vk::BufferCreateInfo{
            size: 3 * size_of::<Vertex>() as vk::DeviceSize,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..vk::BufferCreateInfo::default()
        };

        let vertex_input_buffer = unsafe {
            vk_core
                .device()
                .create_buffer(&vertex_input_buffer_create_info, None)
        }.expect("failed to create vertex input buffer");

        let vertex_input_buffer_memory_requirements = unsafe {
            vk_core.device().get_buffer_memory_requirements(vertex_input_buffer)
        };
        let vertex_input_buffer_memory_index = utils::find_memory_type_index(
            &vertex_input_buffer_memory_requirements,
            device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).expect("failed to find suitable memory type for the vertex input buffer");

        let vertex_input_buffer_memory_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: vertex_input_buffer_memory_requirements.size,
            memory_type_index: vertex_input_buffer_memory_index,
            ..vk::MemoryAllocateInfo::default()
        };

        let vertex_input_buffer_memory = unsafe {
            vk_core
                .device()
                .allocate_memory(&vertex_input_buffer_memory_allocate_info, None)
        }.expect("failed to allocate memory for the vertex input buffer");

        let vert_ptr = unsafe {
            vk_core.device().map_memory(
                vertex_input_buffer_memory,
                0,
                vertex_input_buffer_memory_requirements.size,
                vk::MemoryMapFlags::empty()
            )
        }.expect("failed to map memory");

        let mut vert_slice = unsafe {
            Align::new(
                vert_ptr,
                align_of::<Vertex>() as u64,
                vertex_input_buffer_memory_requirements.size
            )
        };

        vert_slice.copy_from_slice(&vertices);

        unsafe {
            vk_core.device().unmap_memory(vertex_input_buffer_memory);
            vk_core
                .device()
                .bind_buffer_memory(vertex_input_buffer, vertex_input_buffer_memory, 0)
                .expect("failed to bind vertex input buffer memory");
        };

        let vertex_shader_module = shader_loader::load(
            vk_core.device(),
            "vertex_shader",
            "main"
        );

        let fragment_shader_module = shader_loader::load(
            vk_core.device(),
            "fragment_shader",
            "main"
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



        let graphics_pipeline = GraphicsPipeline::new(
            vk_core.clone(),
            renderer,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            vertex_input_state_info,
            shader_stage_create_infos
        );

        Self {
            vk_core,
            vertices,
            indices,
            index_buffer,
            vertex_input_buffer,
            index_buffer_memory,
            vertex_input_buffer_memory,
            graphics_pipeline,
            vertex_shader_module,
            fragment_shader_module,
        }
    }

    pub fn draw(
        &self,
        cmd_buffer: vk::CommandBuffer,
    ) 
    {
        let device = self.vk_core.device();
        unsafe {


            device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.graphics_pipeline(),
            );
            

            device.cmd_bind_vertex_buffers(cmd_buffer, 0, &[self.vertex_input_buffer], &[0]);
            device.cmd_bind_index_buffer(cmd_buffer, self.index_buffer, 0, vk::IndexType::UINT32);

            device.cmd_draw_indexed(cmd_buffer, self.indices.len() as u32, 1, 0, 0, 1);

        }

    }
}

impl Drop for TriangleDrawer {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.free_memory(self.index_buffer_memory, None);
            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.vertex_input_buffer_memory, None);
            device.destroy_buffer(self.vertex_input_buffer, None);
            device.destroy_shader_module(self.vertex_shader_module, None);
            device.destroy_shader_module(self.fragment_shader_module, None);
        }
    }
}
