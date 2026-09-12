use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::descriptors::PipelineLayout;
use crate::backends::vulkan::runtime::swapchain::window_render_target::WindowRenderTarget;
use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;

pub struct GraphicsPipeline {
    vk_context: Arc<VulkanContext>,
    graphics_pipeline: vk::Pipeline,
    pipeline_layout: PipelineLayout,
}

impl GraphicsPipeline {
    pub fn new(
        vk_context: Arc<VulkanContext>,
        render_target: &WindowRenderTarget,
        topology: Option<vk::PrimitiveTopology>,
        vertex_input_state_info: Option<vk::PipelineVertexInputStateCreateInfo>,
        shader_stage_create_infos: Vec<vk::PipelineShaderStageCreateInfo>,
        pipeline_layout: PipelineLayout,
    ) -> Result<Self> {
        let rasterization_state_info = vk::PipelineRasterizationStateCreateInfo::default()
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0)
            .polygon_mode(vk::PolygonMode::FILL);

        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let noop_stencil_state = vk::StencilOpState::default()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .depth_fail_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS);

        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .front(noop_stencil_state)
            .back(noop_stencil_state)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::SRC_COLOR)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_COLOR)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ZERO)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

        let viewport_state_info = render_target.viewport_state_info();

        let topology = topology.unwrap_or_default();

        let vertex_input_assembly_state_info =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);

        let surface_format = render_target
            .surface()
            .get_physical_device_surface_formats(*vk_context.physical_device())?
            .into_iter()
            .next()
            .context("surface returned no supported formats")?;
        let color_attachment_formats = [surface_format.format];
        let depth_attachment_format = render_target.depth_image().format();

        let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(depth_attachment_format);

        let vertex_input_state_info = vertex_input_state_info.unwrap_or_default();

        let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_state_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout.vk_pipeline_layout())
            .push_next(&mut pipeline_rendering_info);

        let graphics_pipeline = unsafe {
            vk_context.device().create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphics_pipeline_create_info],
                None,
            )
        }
        .map_err(|(_, error)| error)?[0];

        Ok(Self {
            vk_context,
            graphics_pipeline,
            pipeline_layout,
        })
    }

    pub fn graphics_pipeline(&self) -> vk::Pipeline {
        self.graphics_pipeline
    }

    pub fn pipeline_layout(&self) -> &PipelineLayout {
        &self.pipeline_layout
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        let device = self.vk_context.device();
        unsafe {
            device.destroy_pipeline(self.graphics_pipeline, None);
        };
    }
}
