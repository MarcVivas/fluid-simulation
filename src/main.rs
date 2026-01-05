mod vk_core;
mod vk_utils;
mod renderer;
mod resources;
mod world;
mod compute;
mod utils;
mod components;
mod systems;
mod physics_engine;

use std::default::Default;
use std::sync::Arc;
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi;
use winit::event::{KeyEvent, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};
use crate::compute::ComputeCommandPool;
use crate::resources::Particles;
use crate::renderer::renderer::Renderer;
use crate::utils::input_manager;
use crate::vk_core::init_with_window;
use crate::vk_core::vk_core::VkCore;
use crate::world::World;

#[allow(unused)]
fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app);
}


struct App {
    vk_core: Option<Arc<VkCore>>,
    renderer: Option<Renderer>,
    window: Option<Window>,
    window_resized: bool,
    world: Option<World>,
    compute_command_pool: Option<ComputeCommandPool>,
    mouse_position: dpi::PhysicalPosition<f64>,
}

impl App {
    pub fn new() -> Self {

        Self {
            window: None,
            vk_core: None,
            renderer: None,
            world: None,
            compute_command_pool: None,
            window_resized: false,
            mouse_position: dpi::PhysicalPosition::default(),
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

      
        self.compute_command_pool = Some(ComputeCommandPool::new(vk_core.clone()).unwrap());
        
        self.world = Some(World::new(
            &vk_core,
            Vec3::new(1000.0, 1000.0, 1000.0),
            self.renderer.as_ref().unwrap(),
            self.compute_command_pool.as_ref().unwrap()
        ));

        self.vk_core = Some(vk_core);
        self.window = Some(window);
        
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                unsafe {
                    self.vk_core.as_ref().unwrap().device().device_wait_idle()
                        .unwrap();
                }
                event_loop.exit();
            },
            WindowEvent::Resized(_logical_size) => {
                self.window_resized = true;
                self.renderer.as_mut().unwrap().resize_window(self.window.as_ref().unwrap());
            }

            WindowEvent::RedrawRequested => {
                self.world.as_mut().unwrap().update(self.vk_core.as_ref().unwrap(),1.0 / 60.0, self.compute_command_pool.as_ref().unwrap());
                self.window.as_ref().unwrap().request_redraw();
                self.renderer.as_mut().unwrap().draw_world(self.window.as_ref().unwrap(), self.world.as_ref().unwrap());
            },
            WindowEvent::KeyboardInput {
                event:
                KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: key_state,
                    ..
                },
                ..
            } => input_manager::process_keyboard_input(self, event_loop, &code, &key_state),
            WindowEvent::CursorMoved { position, .. } => input_manager::process_cursor_moved(self, &position),
            WindowEvent::MouseInput {state: mouse_state, button: mouse_button, ..} => input_manager::process_mouse_input(self, &mouse_state, &mouse_button),
            WindowEvent::MouseWheel { delta, .. } => input_manager::process_mouse_wheel(self, delta),
            _ => (),
        }
    }
    
}


impl App {
    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool){
        self.renderer.as_mut().unwrap().move_camera(key, is_pressed);
    }
    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta){
        self.renderer.as_mut().unwrap().zoom_camera(mouse_scroll_delta);
    }
    
    pub fn set_mouse_position(&mut self, position: Option<dpi::PhysicalPosition<f64>>) {
        self.mouse_position = position.unwrap();
        self.renderer.as_mut().unwrap().set_camera_zoom_position(position);
    }
    
}
