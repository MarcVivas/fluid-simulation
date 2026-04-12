use winit::event_loop::{ControlFlow, EventLoop};
use engine::app::App;


#[allow(unused)]
fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app);
}

