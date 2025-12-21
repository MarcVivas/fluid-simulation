use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode};
use crate::App;

/// Manages keyboard inputs from the user
pub fn process_keyboard_input(app: &mut App, event_loop: &ActiveEventLoop, code: &KeyCode, key_state: &ElementState) {
    match (code, key_state.is_pressed()) {
        (KeyCode::Escape, true) => event_loop.exit(),
        (KeyCode::KeyP, true) => {
            
        },
        (KeyCode::KeyG, true) => {
            
        },
        (KeyCode::KeyW | KeyCode::ArrowUp, true) => {
            app.move_camera(KeyCode::KeyW, true);
        },
        (KeyCode::KeyW | KeyCode::ArrowUp, false) => {
            app.move_camera(KeyCode::KeyW, false);
        },
        (KeyCode::KeyS | KeyCode::ArrowDown, true) => {
            app.move_camera(KeyCode::KeyS, true);
        },
        (KeyCode::KeyS | KeyCode::ArrowDown, false) => {
            app.move_camera(KeyCode::KeyS, false);
        },
        (KeyCode::KeyA | KeyCode::ArrowLeft, true) => {
            app.move_camera(KeyCode::KeyA, true);
        },
        (KeyCode::KeyA | KeyCode::ArrowLeft, false) => {
            app.move_camera(KeyCode::KeyA, false);
        },
        (KeyCode::KeyD | KeyCode::ArrowRight, true) => {
            app.move_camera(KeyCode::KeyD, true);
        },
        (KeyCode::KeyD | KeyCode::ArrowRight, false) => {
            app.move_camera(KeyCode::KeyD, false);
        },
        _ => {}
    }
}

/// Manages mouse movement
pub fn process_cursor_moved(app: &mut App, position: &PhysicalPosition<f64>){
    // Update the stored mouse position
    app.set_mouse_position(Some(*position));
}

/// Manages mouse button inputs from the user
pub fn process_mouse_input(app: &mut App, mouse_state: &ElementState, button: &MouseButton){
}

/// Manages mouse wheel inputs from the user
pub fn process_mouse_wheel(app: &mut App, delta: MouseScrollDelta){
    app.zoom_camera(delta);
}


