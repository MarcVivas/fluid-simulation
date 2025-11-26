use std::path::PathBuf;
use ash::Device;
use ash::util::read_spv;
use ash::vk::ShaderModule;

/// Loads a precompiled shader from the `OUT_DIR`
pub fn load(
    device: &Device,
    shader_name: &str,
) -> ShaderModule
{
    let path = PathBuf::from(env!("OUT_DIR"))
        .join(format!("{}.spv", shader_name));

    let mut spirv_bytes = std::fs::File::open(&path).unwrap();

    let shader_code = read_spv(&mut spirv_bytes)
        .unwrap_or_else(|error| panic!("failed to read shader {} {}", shader_name, error));

    let shader_create_info = ash::vk::ShaderModuleCreateInfo::default()
        .code(&shader_code);
    unsafe {
        device.create_shader_module(
            &shader_create_info,
            None,
        )
    }.expect("failed to create shader module")
}
