//! Backend-independent particle initialization and physics parameters.
pub mod initial_state;
pub mod physics_config;
pub mod presets;

pub use initial_state::ParticleState;
pub use presets::ParticleInitPreset;
