use std::sync::Arc;
use ash::vk;
use crate::renderer::renderer::Renderer;
use crate::vk_core::vk_core::VkCore;

pub struct GraphicsPipeline {
    vk_core: Arc<VkCore>,
    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    pub fn new(
        vk_core: Arc<VkCore>, 
        renderer: &Renderer,
        topology: vk::PrimitiveTopology,
        vertex_input_state_info: vk::PipelineVertexInputStateCreateInfo,
        shader_stage_create_infos: Vec<vk::PipelineShaderStageCreateInfo>,
    ) -> Self 
    {

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
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_state);
        
        let render_target = renderer.render_target();
        
        let viewport_state_info = render_target.viewport_state_info();

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            vk_core
                .device()
                .create_pipeline_layout(&pipeline_layout_create_info, None)
        }.expect("failed to create pipeline layout");

        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology);
        
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
            .layout(pipeline_layout)
            .render_pass(render_target.render_pass());

        let graphics_pipeline = unsafe {
            vk_core
                .device()
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphics_pipeline_create_info],
                    None)
        }.expect("failed to create graphics pipeline")[0];
        
        Self {
            vk_core,
            graphics_pipeline,
            pipeline_layout
        }
    }
    
    pub fn graphics_pipeline(&self) -> vk::Pipeline {
        self.graphics_pipeline
    }
    
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_pipeline(self.graphics_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        };
    }
}
