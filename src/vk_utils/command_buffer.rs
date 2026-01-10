use ash::prelude::VkResult;
use ash::vk;

#[derive(Clone, Copy)]
pub struct CommandBuffer {
    cmd_buffer: vk::CommandBuffer
}

impl CommandBuffer {
    pub fn new(cmd_buffer: vk::CommandBuffer) -> Self {
        Self {
            cmd_buffer
        }
    }
    
    pub fn reset(&self, device: &ash::Device, reset_flags: &vk::CommandBufferResetFlags) -> VkResult<()>{
        unsafe {
            device.reset_command_buffer(self.cmd_buffer, *reset_flags)
        }
    }
    
    pub fn begin_command_buffer(&self, device: &ash::Device, cmd_begin_info: &vk::CommandBufferBeginInfo) -> VkResult<()> {
        unsafe {
            device.begin_command_buffer(self.cmd_buffer, cmd_begin_info)
        }
    }
    
    pub fn end_command_buffer(&self, device: &ash::Device) -> VkResult<()> {
        unsafe {
            device.end_command_buffer(self.cmd_buffer)
        }
    }
    
    pub fn begin_render_pass(&self, device: &ash::Device, rendering_info: &vk::RenderingInfoKHR){
        unsafe {
            device.cmd_begin_rendering(self.cmd_buffer, rendering_info);
        }
    }
    
    pub fn set_viewport(&self, device: &ash::Device, first_viewport: u32, viewports: &[vk::Viewport]){
        unsafe {
            device.cmd_set_viewport(self.cmd_buffer, first_viewport, viewports);
        }
    }
    
    pub fn set_scissor(&self, device: &ash::Device, first_scissor: u32, scissors: &[vk::Rect2D]){
        unsafe {
            device.cmd_set_scissor(self.cmd_buffer, first_scissor, scissors);
        }
    }
    
    pub fn end_rendering(&self, device: &ash::Device){
        unsafe {
            device.cmd_end_rendering(self.cmd_buffer);
        }
    }
    
    pub fn pipeline_barrier2(&self, device: &ash::Device, dependency_info: &vk::DependencyInfoKHR) {
        unsafe{
            device.cmd_pipeline_barrier2(self.cmd_buffer, &dependency_info);
        }
    }
    
    pub fn push_constants(&self, device: &ash::Device, pipeline_layout: vk::PipelineLayout, stage_flags: vk::ShaderStageFlags, offset: u32, push_constants: &[u8]) {
        unsafe {
            device.cmd_push_constants(
                self.cmd_buffer,
                pipeline_layout,
                stage_flags,
                offset,
                push_constants
            );
        }
    }
    
    pub fn vk_cmd_buffer(&self) -> vk::CommandBuffer {
        self.cmd_buffer
    }
    
    pub fn bind_pipeline(&self, device: &ash::Device, bind_info: vk::PipelineBindPoint, pipeline: vk::Pipeline){
        unsafe {
            device.cmd_bind_pipeline(self.cmd_buffer, bind_info, pipeline);
        }
    }
    
    pub fn bind_vertex_buffers(&self, device: &ash::Device, first_binding: u32, buffers: &[vk::Buffer], offsets: &[vk::DeviceSize]){
        unsafe {
            device.cmd_bind_vertex_buffers(self.cmd_buffer, first_binding, buffers, offsets);
        }
    }
    
    pub fn bind_index_buffer(&self, device: &ash::Device, buffer: vk::Buffer, offset: vk::DeviceSize, index_type: vk::IndexType){
        unsafe {
            device.cmd_bind_index_buffer(self.cmd_buffer, buffer, offset, index_type);
        }
    }
    
    pub fn draw_indexed(&self, device: &ash::Device, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32){
        unsafe {
            device.cmd_draw_indexed(self.cmd_buffer, index_count, instance_count, first_index, vertex_offset, first_instance);
        }
    }
    
    pub fn bind_descriptor_sets(&self, device: &ash::Device, pipeline_bind_point: vk::PipelineBindPoint, layout: vk::PipelineLayout, first_set: u32, descriptor_sets: &[vk::DescriptorSet], dynamic_offsets: &[u32]){
        unsafe {
            device.cmd_bind_descriptor_sets(self.cmd_buffer, pipeline_bind_point, layout, first_set, descriptor_sets, dynamic_offsets);
        }
    }
    
    pub fn fill_buffer(&self, device: &ash::Device, buffer: vk::Buffer, offset: vk::DeviceSize, size: vk::DeviceSize, data: u32){
        unsafe {
            device.cmd_fill_buffer(self.cmd_buffer, buffer, offset, size, data);
        }
    }
}