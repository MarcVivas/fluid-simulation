use std::sync::Arc;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{Image, ImageLayout, ImageUsage};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::{single_pass_renderpass, swapchain, sync, Validated, VulkanError};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};
use vulkano::image::view::ImageView;
use vulkano::instance::Instance;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::swapchain::{PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::GpuFuture;
use winit::window::Window;
use crate::shader_loader;
use crate::vk_engine::VkEngine;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Vertex, BufferContents)]
pub struct MyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}


pub struct Renderer {
    surface: Arc<Surface>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    framebuffers: Vec<Arc<Framebuffer>>,
    render_pass: Arc<RenderPass>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    fences: Vec<Option<Arc<FenceSignalFuture<Box<dyn GpuFuture>>>>>,
    previous_fence_idx: usize,
    command_buffer_builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl Renderer {
    pub fn new(
        vk_engine: &VkEngine,
        window: &Arc<Window>,
        surface: Arc<Surface>,       
    ) -> Self 
    {
        let device = vk_engine.device();
        let queue = vk_engine.queue();
        let queue_family_index = vk_engine.queue_family_index();
        
        let (swapchain, swapchain_images) = Self::create_swapchain(
            &device,
            &surface,
            &window,
        );
        
        let memory_allocator = Arc::new(
            StandardMemoryAllocator::new_default(device.clone())
        );

        let vertices = vec![
            MyVertex { position: [-0.5, -0.5] },
            MyVertex { position: [0.5, 0.5] },
            MyVertex { position: [0.5, -0.5] }
        ];

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            },
            vertices
        ).unwrap();

        let command_buffer_allocator = Arc::new(
            StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            )
        );

        let command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue_family_index,
            CommandBufferUsage::MultipleSubmit,
        ).expect("failed to create command buffer builder");
        
        
        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::PresentSrc,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).expect("failed to create render pass");

        let framebuffers = Self::create_framebuffers(
            &swapchain_images,
            &render_pass
        );
        
        let graphics_pipeline = Self::create_graphics_pipeline(
            device,
            window,
            &render_pass
        );

        let command_buffers = Self::get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &graphics_pipeline,
            &framebuffers,
            &vertex_buffer,
        );

        let frames_in_flight = swapchain_images.len();
        let fences = vec![None; frames_in_flight];
        let previous_fence_idx = 0;
        
        Self {
            surface,
            swapchain,
            swapchain_images,
            framebuffers,
            render_pass,
            graphics_pipeline,
            fences,
            previous_fence_idx,
            command_buffer_builder,
            vertex_buffer,
            command_buffers,
            command_buffer_allocator,       
        }
    }

    fn get_command_buffers(
        command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
        queue: &Arc<Queue>,
        pipeline: &Arc<GraphicsPipeline>,
        framebuffers: &Vec<Arc<Framebuffer>>,
        vertex_buffer: &Subbuffer<[MyVertex]>,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        framebuffers.iter().map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            ).unwrap();

            unsafe {
                builder.begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                ).unwrap().bind_pipeline_graphics(pipeline.clone()).unwrap().bind_vertex_buffers(0, vertex_buffer.clone()).unwrap().draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap().end_render_pass(SubpassEndInfo::default()).unwrap();
            }

            builder.build().unwrap()
        }).collect()
    }

    fn create_framebuffers(images: &[Arc<Image>], render_pass: &Arc<RenderPass>)
                           -> Vec<Arc<Framebuffer>> {
        images.iter().map(|image| {
            let view = ImageView::new_default(image.clone()).expect("failed to create image view");
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..FramebufferCreateInfo::default()
                }
            ).expect("failed to create framebuffer")
        }).collect()
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        window: &Arc<Window>,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let vs = shader_loader::load(
            device,
            "vertex_shader",
            "main"
        );
        let fs = shader_loader::load(
            device,
            "fragment_shader",
            "main"
        );

        let vertex_input_state = VertexInputState::new().binding(0, VertexInputBindingDescription {
            stride: std::mem::size_of::<MyVertex>() as u32,
            input_rate: vulkano::pipeline::graphics::vertex_input::VertexInputRate::Vertex,
            ..VertexInputBindingDescription::default()
        }).attribute(0, VertexInputAttributeDescription {
            binding: 0,
            format: Format::R32G32_SFLOAT,
            offset: 0,
            ..VertexInputAttributeDescription::default()
        });

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages).into_pipeline_layout_create_info(device.clone()).unwrap(),
        ).unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                // The stages of our pipeline, we have vertex and fragment stages.
                stages: stages.into_iter().collect(),
                // Describes the layout of the vertex input and how it should behave.
                vertex_input_state: Some(vertex_input_state),
                // Indicate the type of the primitives (the default is a list of triangles).
                input_assembly_state: Some(InputAssemblyState::default()),
                // Set the fixed viewport.
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            }
        ).expect("failed to create graphics pipeline")
    }

    fn create_swapchain(device: &Arc<Device>, surface: &Arc<Surface>, window: &Arc<Window>)
                        -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        let surface_capabilities = device.physical_device().surface_capabilities(&surface, SurfaceInfo::default()).expect("failed to get surface capabilities");

        let (image_format, _) = device.physical_device().surface_formats(&surface, SurfaceInfo::default()).expect("failed to get surface formats").iter().find(|(format, _)| *format == vulkano::format::Format::B8G8R8A8_SRGB).unwrap_or(&device.physical_device().surface_formats(&surface, SurfaceInfo::default()).expect("failed to get surface formats")[0]
        ).clone();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                // Number of images in the swapchain
                min_image_count: surface_capabilities.min_image_count.max(3),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                image_sharing: vulkano::sync::Sharing::Exclusive,
                // V-Sync active with Fifo
                // V-Sync inactive with MailBox or Immediate
                present_mode: PresentMode::Immediate,
                composite_alpha: surface_capabilities.supported_composite_alpha.into_iter().next().unwrap(),
                ..SwapchainCreateInfo::default()
            }
        ).expect("failed to create swapchain")
    }

    pub fn window_resized(&mut self, device: &Arc<Device>, queue: &Arc<Queue>, window: &Arc<Window>) {
        let (new_swapchain, new_images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: window.inner_size().into(),
            ..self.swapchain.create_info()
        }).expect("failed to recreate swapchain");

        self.swapchain = new_swapchain;
        self.swapchain_images = new_images;

        self.framebuffers = Self::create_framebuffers(
            &self.swapchain_images,
            &self.render_pass
        );

        self.graphics_pipeline = Self::create_graphics_pipeline(
            &device,
            &window,
            &self.render_pass,
        );

        self.command_buffers = Self::get_command_buffers(
            &self.command_buffer_allocator,
            &queue,
            &self.graphics_pipeline,
            &self.framebuffers,
            &self.vertex_buffer
        );
    }

    pub fn draw(&mut self, device: &Arc<Device>, queue: &Arc<Queue>) -> bool {

        // Acquire image to draw
        let (image_idx, suboptimal, acquire_future) = match swapchain::acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
            Ok(res) => res,
            Err(VulkanError::OutOfDate) => {
                // recreate swapchain
                return true;
            }
            Err(e) => panic!("failed to acquire next image: {:?}", e),
        };

        if suboptimal {
            // Recreate swapchain
            return true;
        }

        // Only wait for the fence related to this image to finish
        if let Some(image_fence) = &self.fences[image_idx as usize] {
            // Still using the image, wait for it to finish
            image_fence.wait(None).unwrap();
        }

        let prev_future = match self.fences[self.previous_fence_idx].clone() {
            // Create a "NowFuture". 
            None => {
                let mut now = sync::now(device.clone());
                now.cleanup_finished();
                now.boxed()
            }
            // Use the existing fence
            Some(fence) => fence.boxed(),
        };

        let future = prev_future.join(acquire_future).then_execute(
            queue.clone(),
            self.command_buffers[image_idx as usize].clone()
        ).unwrap().then_swapchain_present(
            queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(
                self.swapchain.clone(),
                image_idx
            )
        ).boxed().then_signal_fence_and_flush();

        // Substitute the previous fence with the new one
        self.fences[image_idx as usize] = match future.map_err(Validated::unwrap) {
            Ok(val) => Some(Arc::new(val)),
            Err(VulkanError::OutOfDate) => {
                // Recreate swapchain
                return true;
            }
            Err(e) => {
                panic!("failed to flush future: {:?}", e);
            }
        };

        self.previous_fence_idx = image_idx as usize;

        return false;
    }
}


