//! Logical world data, independent of GPU APIs.
pub mod bounds;
pub mod particles;
mod world;

pub use bounds::WorldBounds;
pub use world::World;
