mod vk_core;
mod shader_loader;
mod renderer;
mod utils;

use std::default::Default;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};
use crate::renderer::renderer::Renderer;
use crate::renderer::triangle_drawer::TriangleDrawer;
use crate::vk_core::init_with_window;
use crate::vk_core::vk_core::VkCore;

#[allow(unused)]
fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app);
}


struct App {
    window: Option<Window>,
    vk_core: Option<Arc<VkCore>>,
    renderer: Option<Renderer>,
    window_resized: bool,
    recreate_swapchain: bool,
}

impl App {
    pub fn new() -> Self {
       
        Self {
            window: None,
            vk_core: None,
            renderer: None,
            window_resized: false,
            recreate_swapchain: false,
        }
    }
}

impl ApplicationHandler for App {
    
    /// This creates the window and the engine before the event loop starts.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = {
            let window_attributes = WindowAttributes::default()
                .with_title("Vulkan")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
            event_loop.create_window(window_attributes)
                .expect("Failed to create window")
        };


        let (vk_core, surface) = init_with_window(&window);

        self.renderer = Some(
            Renderer::new(
                vk_core.clone(),
                &window,
                surface
            )
        );
        let triangle_drawer = TriangleDrawer::new(vk_core.clone(), self.renderer.as_ref().unwrap());
        self.renderer.as_mut().unwrap().set_triangle_drawer(triangle_drawer);
        
        self.vk_core = Some(vk_core);
        self.window = Some(window);

    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::Resized(logical_size) => {
                self.window_resized = true;
                self.renderer.as_mut().unwrap().resize_window(self.window.as_ref().unwrap());
            }

            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                // Draw.
                
                self.renderer.as_mut().unwrap().draw(self.window.as_ref().unwrap());

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw in
                // applications which do not always need to. Applications that redraw continuously
                // can render here instead.
                self.window.as_ref().unwrap().request_redraw();

               
            }
            _ => (),
        }
    }
    
    
} 



