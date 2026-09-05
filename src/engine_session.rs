use std::sync::{Arc};

use anyhow::{Context, Ok, Result};
use ash::vk;
use winit::{dpi::PhysicalPosition, event::{ElementState, MouseButton, MouseScrollDelta}, keyboard::KeyCode, window::Window};

use crate::{rendering::{simulation_visualizer::SimulationVisualizer}, simulation::simulation::Simulation, vulkan::{core::VulkanContext, core::surface::Surface, frame::frame_pacer::FramePacer, profiler::GpuProfiler}};

pub struct EngineSession {
    vk_core: Arc<VulkanContext>,
    simulation_visualizer: SimulationVisualizer,
    simulation: Simulation,
    gpu_profiler: GpuProfiler,
    frame_pacer: FramePacer,

}

impl EngineSession {
     pub fn new(vk_core: Arc<VulkanContext>, window: &Window, surface: Surface) -> Result<Self> {
         let world_size = glam::Vec3::new(256.0, 256.0, 256.0);
         let frame_pacer = FramePacer::new(3);
         let frames_in_flight = frame_pacer.frames_in_flight();

         let (render_sender, render_receiver) = crossbeam_channel::bounded(1);

         let simulation = Simulation::new(vk_core.clone(), world_size, frames_in_flight, render_sender).context("Failed to create Simulation")?;
         let simulation_visualizer = SimulationVisualizer::new(vk_core.clone(), window, surface, &world_size, render_receiver, frames_in_flight)
            .context("Failed to create SimulationVisualizer")?;
         let gpu_profiler = GpuProfiler::new(vk_core.clone(), 100, frames_in_flight)
            .context("Failed to create GPU profiler")?;

         simulation.send_render_data().context("failed to send initial render data")?;


         Ok(
             Self {
                 vk_core,
                 simulation,
                 simulation_visualizer,
                 gpu_profiler,
                 frame_pacer,
             }
         )
     }

     /// Updates and renders the simulation
     pub fn update_and_render(&mut self, window: &Window) -> Result<()> {
         // Begin Frame (Check constraints, acquire image)
        if !self.simulation_visualizer.begin_frame(window, &self.frame_pacer).context("Couldn't begin frame")? {
            return Ok(()); // Out of date, minimized, etc.
        }

         let device = self.vk_core.device();

         self.gpu_profiler.print_metrics(device, &self.frame_pacer);
         self.gpu_profiler.reset_on_host(device, self.frame_pacer.ring_index());


         let compute_finished_semaphore = self.simulation.compute_finished_semaphore(&self.frame_pacer);

         // Run the simulation and the visualizer in parallel
         let (sim_result, vis_result) = rayon::join(
             || self.simulation.update(&self.vk_core, &self.frame_pacer, &self.gpu_profiler),
             || self.simulation_visualizer.update(&self.frame_pacer),
         );
         sim_result.context("simulation update failed")?;
         vis_result.context("visualizer update failed")?;

         // Submit the work to the GPU!
         self.submit_to_queue(compute_finished_semaphore)?;

         self.frame_pacer.advance();

         Ok(())
     }

     /// Submits the commands to the GPU
     fn submit_to_queue(&mut self, compute_finished_semaphore: vk::SemaphoreSubmitInfo) -> Result<()>{
        self.simulation.submit_update(&self.frame_pacer)?;
        self.simulation_visualizer.submit_and_present(compute_finished_semaphore, &self.frame_pacer)?;
        Ok(())
     }

     pub fn resize_window(&mut self, window: &Window) -> Result<()>{
         self.simulation_visualizer.resize_window(window, &self.frame_pacer)
     }

     pub fn close(&self,) -> Result<()>{
         unsafe {
             self.vk_core
                 .device()
                 .device_wait_idle()

         }.context("Failed to wait for active GPU queues to idle during session shutdown")?;
         Ok(())
     }

     pub fn toggle_pause(&mut self) {
         self.simulation.toggle_pause();
     }

     
     pub fn handle_key_input(&mut self, code: KeyCode, is_pressed: bool) {
         self.simulation_visualizer.camera_controller().handle_key_input(code, is_pressed);
     }
 
     pub fn set_mouse_position(&mut self, position: Option<PhysicalPosition<f64>>) {
         self.simulation_visualizer.camera_controller().set_mouse_position(position);
     }
 
     pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
         self.simulation_visualizer.camera_controller().handle_rotation(&button, &state);
     }
 
     pub fn handle_zoom(&mut self, delta: MouseScrollDelta) {
         self.simulation_visualizer.camera_controller().handle_zoom(delta);
     }
}

impl Drop for EngineSession {
    fn drop(&mut self) {
        if let Err(err) = self.close() {
            eprintln!("Error during EngineSession shutdown: {:?}", err);
        }   
    }
}
