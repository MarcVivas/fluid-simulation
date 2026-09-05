use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode};
use crate::engine_session::EngineSession;
use winit::event::{WindowEvent,};
use winit::keyboard::{PhysicalKey};

pub fn handle_input(event: &WindowEvent, event_loop: &ActiveEventLoop, session: &mut EngineSession) {
    match event {
        WindowEvent::KeyboardInput { event, .. } => {
            if let PhysicalKey::Code(code) = event.physical_key {
                let is_pressed = event.state.is_pressed();
                if is_pressed {
                    match code {
                        KeyCode::Escape => { event_loop.exit(); return; }
                        KeyCode::Space => { session.toggle_pause(); return; }
                        _ => {}
                    }
                }
                session.handle_key_input(code, is_pressed);
            }
        }
        WindowEvent::CursorMoved { position, .. } => session.set_mouse_position(Some(*position)),
        WindowEvent::MouseInput { state, button, .. } => session.handle_mouse_button(*button, *state),
        WindowEvent::MouseWheel { delta, .. } => session.handle_zoom(*delta),
        _ => {}
    }
}
