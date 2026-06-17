use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::vulkan::vk_utils::shader_constants::ShaderCompileTimeConstants;

/// Recursively searches for the slangc compiler binary inside the local SDK directory
fn find_slangc_executable(sdk_dir: &Path) -> Option<PathBuf> {
    let target_name = if cfg!(target_os = "windows") { "slangc.exe" } else { "slangc" };
    find_file_recursive(sdk_dir, target_name)
}

fn find_file_recursive(dir: &Path, target_name: &str) -> Option<PathBuf> {
    fs::read_dir(dir).ok()?.flatten().find_map(|entry| {
        let path = entry.path();
        if path.is_dir() {
            find_file_recursive(&path, target_name)
        } else if path.file_name()?.to_str()? == target_name {
            Some(path)
        } else {
            None
        }
    })
}

/// Runtime shader compilation using the local slangc executable
pub fn get_shader_bytecode(file_name: &str, constants: &ShaderCompileTimeConstants) -> Vec<u32> {
    println!("Compiling shader: {:?}", file_name);

    // Resolve directories and paths
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let project_root = PathBuf::from(manifest_dir);
    let shader_dir = project_root.join("src/shaders");

    let original_path = shader_dir.join(format!("{}.slang", file_name));
    
    let temp_name = format!("{}_dynamic", file_name);
    let temp_input_path = shader_dir.join(format!("{}.slang", temp_name));
    let temp_output_path = shader_dir.join(format!("{}.spv", temp_name));

    // Read and generate the dynamic source code
    let original_source = fs::read_to_string(&original_path)
        .unwrap_or_else(|_| panic!("Failed to read original shader: {:?}", original_path));
    
    let dynamic_source = format!("{}\n{}", constants.as_source_code_header(), original_source);

    // Write the temporary input file
    fs::write(&temp_input_path, &dynamic_source)
        .unwrap_or_else(|_| panic!("Failed to write temp file: {:?}", temp_input_path));

    // Locate the local slangc compiler
    let sdk_dir = project_root.join("target/slang-sdk");
    let slangc_path = find_slangc_executable(&sdk_dir)
        .expect("Slang compiler executable (slangc) could not be found in .slang-sdk");

    // Execute the compilation subprocess
    let output = Command::new(&slangc_path)
        .arg(&temp_input_path)
        .arg("-target")
        .arg("spirv")
        .arg("-matrix-layout-row-major") 
        .arg("-O1")                      
        .arg("-o")
        .arg(&temp_output_path)
        .output()
        .expect("Failed to execute slangc compiler process");

    // Clean up the temporary input file immediately
    let _ = fs::remove_file(&temp_input_path);

    // Check for compiler errors
    if !output.status.success() {
        let error_message = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_file(&temp_output_path); // Clean up output if partially written
        panic!("Slang Compilation Failed for {}:\n{}", file_name, error_message);
    }

    // Read the compiled SPIR-V bytes
    let spirv_bytes = fs::read(&temp_output_path)
        .unwrap_or_else(|_| panic!("Failed to read compiled SPIR-V output: {:?}", temp_output_path));

    // Clean up the temporary output file
    let _ = fs::remove_file(&temp_output_path);

    // Convert bytes to u32 words safely, avoiding potential alignment panics
    spirv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}