use crate::engine_session::{EngineSession};
use crate::vulkan::core::init_with_window;
use anyhow::Context;
use winit::application::ApplicationHandler;
use winit::{dpi};
use winit::event::{WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};
use crate::input_manager; 

pub struct App {
    window: Option<Window>,
    session: Option<EngineSession>,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            session: None,
        }
    }
}

impl ApplicationHandler for App {
    /// This creates the window and the engine before the event loop starts.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = {
            let window_attributes = WindowAttributes::default()
                .with_title("Vulkan")
                .with_inner_size(dpi::LogicalSize::new(1280.0, 720.0));
            event_loop
                .create_window(window_attributes)
                .expect("Failed to create window")
        };

        let (vk_core, surface) = init_with_window(&window)
            .expect("failed to initialize Vulkan context");
        let session = EngineSession::new(vk_core, &window, surface).expect("failed to initialize engine session");
        
        self.window = Some(window);
        self.session = Some(session);
    }

    /// Infinite loop
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(session), Some(window)) = (&mut self.session, &self.window) else {
            return;
        };
      
        match event {
            WindowEvent::CloseRequested => {
                if let Err(err) = session.close() {
                    eprintln!("Error during engine shutdown: {:?}", err);
                }
                event_loop.exit();
            }
            WindowEvent::Resized(_logical_size) => {
                if let Err(err) =  session.resize_window(window).context("Couldn't resize the window") {
                    eprintln!("Error while resizing the window: {:?}", err);
                }                  
            }

            WindowEvent::RedrawRequested => {
                if let Err(err) = session.update_and_render(window) {
                    eprintln!("Error during update_and_render: {:?}", err);
                }
                window.request_redraw();                                    
            },
            _ => (),
        }

        input_manager::handle_input(&event, event_loop, session);
    }

    
}
