use std::fmt::format;
use std::path::PathBuf;
use std::sync::Arc;
use ash::Device;
use ash::util::read_spv;
use ash::vk::ShaderModule;

/// Loads a precompiled shader from the `OUT_DIR`
pub fn load(
    device: &Device,
    shader_name: &str,
    entry_point_name: &str,
) -> ShaderModule
{
    let path = PathBuf::from(env!("OUT_DIR"))
        .join(format!("{}.spv", shader_name));

    let mut spirv_bytes = std::fs::File::open(&path).unwrap();

    let shader_code = read_spv(&mut spirv_bytes)
        .expect(format!("failed to read shader {}", shader_name).as_str());

    let shader_create_info = ash::vk::ShaderModuleCreateInfo::default()
        .code(&shader_code);
    unsafe {
        device.create_shader_module(
            &shader_create_info,
            None,
        )
    }.expect("failed to create shader module")
}