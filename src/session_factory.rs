use anyhow::{Context, Result};
use winit::window::Window;

use crate::app_session::AppSession;
use crate::backends::BackendKind;
use crate::backends::gpu_backend::GpuBackend;
use crate::backends::vulkan::VulkanBackend;
use crate::backends::vulkan::runtime::core::init_with_window;
use crate::session::Session;
use crate::world::World;

pub fn create_session(
    window: &Window,
    backend_kind: BackendKind,
    world: World,
) -> Result<Box<dyn AppSession>> {
    let backend: Box<dyn GpuBackend> = match backend_kind {
        BackendKind::Vulkan => {
            let (vk_context, surface) =
                init_with_window(window).context("Failed to initialize Vulkan context")?;
            Box::new(
                VulkanBackend::new(vk_context, window, surface, &world)
                    .context("Failed to initialize Vulkan backend")?,
            )
        }
    };
    let session =
        Session::new(backend, window, world).context("Failed to initialize particle session")?;
    Ok(Box::new(session))
}
