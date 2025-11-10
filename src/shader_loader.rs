use std::path::PathBuf;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::shader::{EntryPoint, ShaderModule, ShaderModuleCreateInfo};

/// Loads a precompiled shader from the `OUT_DIR`
pub fn load(
    device: &Arc<Device>,
    shader_name: &str,
    entry_point_name: &str,
) -> EntryPoint 
{
    let path = PathBuf::from(env!("OUT_DIR"))
        .join(format!("{}.spv", shader_name));

    let spirv_bytes: Vec<u8> = std::fs::read(&path).unwrap();
    let spirv_u32 = convert_bytes_to_u32(&spirv_bytes);
    
    let vs = unsafe {
        ShaderModule::new(
            device.clone(),
            ShaderModuleCreateInfo::new(&spirv_u32)
        )
    }.expect("failed to create shader module");

    vs.entry_point(entry_point_name).unwrap()
}

fn convert_bytes_to_u32(bytes: &[u8]) -> &[u32] {
    assert_eq!(bytes.len() % 4, 0);
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const u32,
            bytes.len() / 4
        )
    }
}