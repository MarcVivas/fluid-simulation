use std::path::PathBuf;

use crate::vulkan::vk_utils::shader_constants::ShaderCompileTimeConstants;

use shader_slang::{
    Downcast
};


/// Runtime shader compilation
#[cfg(debug_assertions)]
pub fn get_shader_bytecode(file_name: &str, constants: &ShaderCompileTimeConstants) -> Vec<u32> {
    use std::fs;
    use shader_slang::{GlobalSession, SessionDesc, CompilerOptions, TargetDesc, CompileTarget};
    use std::ffi::CString;

    // 1. Get the absolute path to the project directory
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
        
    // 2. Build the absolute path to the shaders folder
    let mut shader_dir = PathBuf::from(manifest_dir);
    shader_dir.push("src");
    shader_dir.push("shaders");
    
    // 3. Resolve the original file path
    let mut original_path = shader_dir.clone();
    original_path.push(format!("{}.slang", file_name));
    
    let original_source = fs::read_to_string(&original_path)
        .unwrap_or_else(|_| panic!("Failed to read original shader: {:?}", original_path));
    
    let dynamic_source = format!("{}\n{}", constants.as_source_code_header(), original_source);
    
    // 4. Resolve the temp file path
    let temp_name = format!("{}_dynamic", file_name);
    let mut temp_path = shader_dir.clone();
    temp_path.push(format!("{}.slang", temp_name));
    
    fs::write(&temp_path, &dynamic_source).unwrap_or_else(|_| panic!("Failed to write temp file: {:?}", temp_path));
    
    // 5. Pass the ABSOLUTE path to Slang's search path
    let shader_dir_str = shader_dir.to_str().unwrap();
    let search_path_cstr = CString::new(shader_dir_str).unwrap();
    
    let global_session = GlobalSession::new().unwrap();
        
    let session_options = CompilerOptions::default()
        .vulkan_use_entry_point_name(true)
        .matrix_layout_row(true);
            
    let target_desc = [TargetDesc::default().format(CompileTarget::Spirv)];
        
    let search_paths = [search_path_cstr.as_ptr()];
        
    let session_desc = SessionDesc::default()
        .targets(&target_desc)
        .search_paths(&search_paths)
        .options(&session_options);
    
    let session = global_session.create_session(&session_desc).unwrap();
        
    // Load the module by its name
    let module_result = session.load_module(&temp_name);
    
    // CRITICAL: Always delete the temp file immediately BEFORE calling expect()
    // so it doesn't get left behind if the compilation fails.
    let _ = fs::remove_file(&temp_path);
    
    // Check for compilation errors
    let module = module_result.expect(&format!("Slang Compilation Failed for {}", temp_name));
    
    let mut components = vec![module.downcast().clone()];
    for i in 0..module.entry_point_count() {
        components.push(module.entry_point_by_index(i).unwrap().downcast().clone());
    }
    
    let program = session.create_composite_component_type(&components).unwrap();
    let linked = program.link().unwrap();
    let spirv_bytes = linked.target_code(0).unwrap();
    
    bytemuck::cast_slice(spirv_bytes.as_slice()).to_vec()
}



/// Loads a precompiled shader from the `OUT_DIR`
#[cfg(not(debug_assertions))]
pub fn get_shader_bytecode(
    shader_name: &str,
    _compile_time_constants: &ShaderCompileTimeConstants
) -> Vec<u32>
{
    use ash::util::read_spv;

    let path = PathBuf::from(env!("OUT_DIR"))
        .join(format!("{}.spv", shader_name));

    let mut spirv_bytes = std::fs::File::open(&path).unwrap();

    let shader_code = read_spv(&mut spirv_bytes)
        .unwrap_or_else(|error| panic!("failed to read shader {} {}", shader_name, error));
    
    shader_code
}
