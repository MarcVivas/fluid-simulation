mod vk_engine;
mod shader_loader;
mod renderer;

use std::default::Default;
use std::sync::Arc;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};
use crate::renderer::Renderer;
use crate::vk_engine::VkEngine;

#[allow(unused)]
fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app);
}


struct App {
    window: Option<Arc<Window>>,
    engine: Option<VkEngine>,
    renderer: Option<Renderer>,
    window_resized: bool,
    recreate_swapchain: bool,
}

impl App {
    pub fn new() -> Self {
       
        Self {
            window: None,
            engine: None,
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
            Arc::new(event_loop.create_window(window_attributes)
                         .expect("Failed to create window")
            )
        };



        let instance = {
            let library = VulkanLibrary::new()
                .expect("no local Vulkan library/DLL");
            let required_extensions = Surface::required_extensions(&event_loop)
                .expect("failed to retrieve required extensions");

            Instance::new(
                library,
                InstanceCreateInfo {
                    flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                    enabled_extensions: required_extensions,
                    ..Default::default()
                },
            ).expect("failed to create instance")
        };

        let surface = Surface::from_window(instance.clone(), window.clone())
            .expect("failed to create surface");
        
        self.engine = Some(VkEngine::new(
            &instance,
            Some(&surface)
        ));
        
        
        
        self.renderer = Some(Renderer::new(
            &self.engine.as_ref().unwrap(),
            &window,
            surface
        ));

        self.window = Some(window);
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::Resized(_logical_size) => {
                self.window_resized = true;
            }

            WindowEvent::RedrawRequested => {
                if self.window_resized || self.recreate_swapchain {
                    self.renderer
                        .as_mut()
                        .expect("Engine not initialized")
                        .window_resized(
                            self.engine.as_ref().unwrap().device(),
                            self.engine.as_ref().unwrap().queue(),
                            self.window.as_ref().expect("Window not initialized")
                        );
                    self.window_resized = false;
                    self.recreate_swapchain = false;
                }
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                // Draw.
                self.recreate_swapchain = self.renderer.as_mut().unwrap().draw(
                    self.engine.as_ref().unwrap().device(),
                    self.engine.as_ref().unwrap().queue()
                );


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



