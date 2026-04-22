use crate::compute::ComputeEngine;
use crate::renderer::renderer::Renderer;
use crate::utils::gpu_profiler::GpuProfiler;
use crate::utils::input_manager;
use crate::vulkan::vk_core::init_with_window;
use crate::vulkan::vk_core::VkCore;
use crate::world::World;
use glam::Vec3;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;
use winit::window::{Window, WindowAttributes, WindowId};
use winit::event::KeyEvent;
use winit::keyboard::PhysicalKey;

pub struct App {
    vk_core: Option<Arc<VkCore>>,
    renderer: Option<Renderer>,
    window: Option<Window>,
    window_resized: bool,
    paused: bool,
    world: Option<World>,
    compute_engine: Option<ComputeEngine>,
    gpu_profiler: Option<GpuProfiler>,
    mouse_position: dpi::PhysicalPosition<f64>,
    total_frames_proccessed: u64
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            vk_core: None,
            renderer: None,
            world: None,
            gpu_profiler: None,
            window_resized: false,
            paused: true,
            mouse_position: dpi::PhysicalPosition::default(),
            compute_engine: None,
            total_frames_proccessed: 0
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

        let (vk_core, surface) = init_with_window(&window);

        let world_size = Vec3::new(256.0, 256.0, 256.0);

        self.renderer = Some(Renderer::new(
            vk_core.clone(),
            &window,
            surface,
            &world_size,
        ));

        let frames_in_flight = Renderer::frames_in_flight();
        self.compute_engine = Some(
            ComputeEngine::new(vk_core.clone(), frames_in_flight)
                .unwrap()
        );
        
        let max_zones = 100;
        self.gpu_profiler = Some(GpuProfiler::new(vk_core.clone(), max_zones, frames_in_flight));

        self.world = Some(World::new(
            &vk_core,
            &world_size,
            self.renderer.as_ref().unwrap(),
        ));

        self.vk_core = Some(vk_core);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                unsafe {
                    self.vk_core
                        .as_ref()
                        .unwrap()
                        .device()
                        .device_wait_idle()
                        .unwrap();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(_logical_size) => {
                self.window_resized = true;
                self.renderer
                    .as_mut()
                    .unwrap()
                    .resize_window(self.window.as_ref().unwrap());
            }

            WindowEvent::RedrawRequested => {
                // Wait for GPU and get safe indices
                let Some((current_frame_idx, image_index)) = self.renderer.as_mut().unwrap().begin_frame(self.window.as_ref().unwrap()) else {
                        return; // Minimized or out of date, skip this frame
                };




                let vk_core = self.vk_core.as_ref().unwrap();
                let compute_engine = self.compute_engine.as_mut().unwrap();
                let renderer = self.renderer.as_mut().unwrap();
                let world = self.world.as_mut().unwrap();
                
                // Update the current frame index from the compute engine
                compute_engine.set_frame_index(current_frame_idx);
                self.gpu_profiler.as_mut().unwrap().set_frame_index(current_frame_idx, self.total_frames_proccessed);
                
                let gpu_profiler = self.gpu_profiler.as_ref().unwrap();
                let timings = gpu_profiler
                    .get_results(vk_core.device(), self.total_frames_proccessed)
                    .unwrap_or_default();
                for (label, time) in timings {
                    if time != 0.0 {
                        println!("Pass {}: {:.4} ms", label, time);
                    }
                }
                
                // Reset query pool
                self.gpu_profiler.as_ref().unwrap().reset_on_host(vk_core.device());
                
                
                
                
                // Extract the rendering data
                let render_data = world.extract_render_data();

                // Record commands to the gpu
                rayon::join(
                    ||{
                        if !self.paused {
                            world.update(vk_core, compute_engine, self.gpu_profiler.as_ref().unwrap());
                        }
                    },
                    ||{
                        renderer.record_draw_commands(
                            &render_data,
                            image_index,
                            current_frame_idx,
                            self.paused
                        );
                    }
                );


                // Submit phase
                if !self.paused {
                    // Submit commands to the queue
                    compute_engine.submit_to_queue(&[]);
                    self.total_frames_proccessed+=1;
                }

                self.window.as_ref().unwrap().request_redraw();
                self.renderer.as_mut().unwrap().submit_and_present(
                        self.compute_engine.as_ref().unwrap().compute_finished_semaphore(self.paused),
                        current_frame_idx,
                        image_index
                );
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => input_manager::process_keyboard_input(self, event_loop, &code, &key_state),
            WindowEvent::CursorMoved { position, .. } => {
                input_manager::process_cursor_moved(self, &position)
            }
            WindowEvent::MouseInput {
                state: mouse_state,
                button: mouse_button,
                ..
            } => input_manager::process_mouse_input(self, &mouse_state, &mouse_button),
            WindowEvent::MouseWheel { delta, .. } => {
                input_manager::process_mouse_wheel(self, delta)
            }
            _ => (),
        }
    }
}

impl App {
    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool) {
        self.renderer.as_mut().unwrap().move_camera(key, is_pressed);
    }
    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta) {
        self.renderer
            .as_mut()
            .unwrap()
            .zoom_camera(mouse_scroll_delta);
    }

    pub fn set_mouse_position(&mut self, position: Option<dpi::PhysicalPosition<f64>>) {
        self.mouse_position = position.unwrap();
        self.renderer
            .as_mut()
            .unwrap()
            .set_camera_zoom_position(position);
    }

    pub fn mouse_click(&mut self, button: &MouseButton, state: &ElementState) {
        self.renderer.as_mut().unwrap().rotate_camera(button, state);
    }

    pub fn toggle_paused(&mut self) {
        self.paused = !self.paused;
    }
}
