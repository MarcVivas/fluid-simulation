use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::vulkan::shaders::ShaderCompileTimeConstants;

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

/// Recursively gathers all directories that contain at least one `.slang` file
fn collect_shader_directories(dir: &Path, shader_dirs: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        let mut contains_shader = false;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_shader_directories(&path, shader_dirs);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("slang") {
                contains_shader = true;
            }
        }
        if contains_shader {
            shader_dirs.push(dir.to_path_buf());
        }
    }
}

/// Runtime shader compilation with disk-based caching
pub fn get_shader_bytecode(file_name: &str, constants: &ShaderCompileTimeConstants) -> Vec<u32> {
    // Resolve basic directories
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let project_root = PathBuf::from(manifest_dir);
    let src_dir = project_root.join("src");

    // Locate the original shader file recursively to read its contents
    let shader_filename = format!("{}.slang", file_name);
    let original_path = find_file_recursive(&src_dir, &shader_filename)
        .unwrap_or_else(|| panic!("Failed to locate shader file '{}' inside {:?}", shader_filename, src_dir));

    let original_source = fs::read_to_string(&original_path)
        .unwrap_or_else(|_| panic!("Failed to read original shader: {:?}", original_path));
    
    let constants_header = constants.as_source_code_header();

    // Normalize the constants header to neutralize HashMap iteration order differences.
    let mut header_lines: Vec<&str> = constants_header.lines().collect();
    header_lines.sort_unstable();
    let normalized_header = header_lines.join("\n");

    // Generate a unique hash
    let mut hasher = DefaultHasher::new();
    file_name.hash(&mut hasher);
    original_source.hash(&mut hasher);
    normalized_header.hash(&mut hasher);
    let hash_val = hasher.finish();

    // Define cache directory inside Cargo's target folder
    let cache_dir = project_root.join("target").join("shader-cache");
    let safe_file_name = file_name.replace("/", "_");
    let cached_spv_path = cache_dir.join(format!("{}_{:x}.spv", safe_file_name, hash_val));

    // Try to load from cache
    if cached_spv_path.exists() {
        if let Ok(spirv_bytes) = fs::read(&cached_spv_path) {
            return spirv_bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
        }
    }

    // --- Cache Miss: Perform Compilation ---
    println!("Compiling shader (Cache Miss): {:?}", file_name);

    let shader_dir = original_path.parent()
        .expect("Failed to get parent directory of the shader");

    // Collect search paths only when compiling
    let mut shader_search_paths = Vec::new();
    collect_shader_directories(&src_dir, &mut shader_search_paths);

    // Setup temporary paths inside the shader directory
    let temp_name = format!("{}_dynamic", safe_file_name);
    let temp_input_path = shader_dir.join(format!("{}.slang", temp_name));
    let temp_output_path = shader_dir.join(format!("{}.spv", temp_name));

    // Combine constants and original source (using the raw, original constants_header for actual compilation)
    let dynamic_source = format!("{}\n{}", constants_header, original_source);
    fs::write(&temp_input_path, &dynamic_source)
        .unwrap_or_else(|_| panic!("Failed to write temp file: {:?}", temp_input_path));

    // Locate the compiler
    let sdk_dir = project_root.join("target/slang-sdk");
    let slangc_path = find_slangc_executable(&sdk_dir)
        .expect("Slang compiler executable (slangc) could not be found in target/slang-sdk");

    let mut command = Command::new(&slangc_path);
    command
        .arg(&temp_input_path)
        .arg("-target")
        .arg("spirv")
        .arg("-matrix-layout-row-major") 
        .arg("-O1");
    
    for path in &shader_search_paths {
        command.arg("-I").arg(path);
    }

    let output = command
        .arg("-o")
        .arg(&temp_output_path)
        .output()
        .expect("Failed to execute slangc compiler process");

    // Clean up temporary input file
    let _ = fs::remove_file(&temp_input_path);

    if !output.status.success() {
        let error_message = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_file(&temp_output_path);
        panic!("Slang Compilation Failed for {}:\n{}", file_name, error_message);
    }

    // Read the compiled binary
    let spirv_bytes = fs::read(&temp_output_path)
        .unwrap_or_else(|_| panic!("Failed to read compiled SPIR-V output: {:?}", temp_output_path));

    // Clean up the temporary compiler output
    let _ = fs::remove_file(&temp_output_path);

    // Save the compiled output to our persistent cache with explicit error checking
    fs::create_dir_all(&cache_dir)
        .unwrap_or_else(|e| panic!("Failed to create shader cache directory {:?}: {:?}", cache_dir, e));
    
    fs::write(&cached_spv_path, &spirv_bytes)
        .unwrap_or_else(|e| panic!("Failed to write cached SPV to {:?}: {:?}", cached_spv_path, e));

    spirv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}