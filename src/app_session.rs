use anyhow::Result;
use winit::{event::WindowEvent, window::Window};

pub trait AppSession {
    fn frame(&mut self, window: &Window) -> Result<()>;
    fn resize(&mut self, window: &Window) -> Result<()>;
    fn shutdown(&mut self) -> Result<()>;
    fn handle_input(&mut self, event: &WindowEvent);
}
