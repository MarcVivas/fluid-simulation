use anyhow::Result;
use winit::window::Window;

use crate::viewer::camera::CameraUniform;

pub trait GpuBackend {
    /// Returns None when this frame should be skipped
    fn begin_frame(&mut self, window: &Window) -> Result<Option<glam::Vec2>>;

    /// Call once after begin frame retunrs Some
    fn execute_frame(&mut self, camera: &CameraUniform, paused: bool) -> Result<()>;

    /// For window resizing
    fn resize(&mut self, window: &Window) -> Result<()>;

    /// Use this for when you are closing the program
    fn shutdown(&self) -> Result<()>;
}
