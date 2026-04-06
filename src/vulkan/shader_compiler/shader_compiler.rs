use shader_slang::{
    CompilerOptions,
    GlobalSession,
    TargetDesc,
    SessionDesc,
    Downcast,
    CompileTarget
};

use std::ffi::CString;
use std::fs;

use crate::vulkan::shader_compiler::shader_constants::ShaderCompileTimeConstants;

/// A generic runtime compiler for your Slang files
pub fn compile_slang_kernel(file_name: &str, defines: &ShaderCompileTimeConstants) -> Vec<u8> {
    let global_session = GlobalSession::new().unwrap();
    let search_path = CString::new("src/shaders").unwrap();

    let session_options = CompilerOptions::default()
        .vulkan_use_entry_point_name(true)
        .matrix_layout_row(true);

    let target_desc = [TargetDesc::default().format(CompileTarget::Spirv)];
    let search_paths = [search_path.as_ptr()];

    let session_desc = SessionDesc::default()
        .targets(&target_desc)
        .search_paths(&search_paths)
        .options(&session_options);

    // 1. Read the original shader code
    let original_path = format!("src/shaders/{}.slang", file_name);
    let original_source = fs::read_to_string(&original_path)
        .expect(&format!("Failed to read shader: {}", original_path));

    // 2. Prepend the dynamic #define macros to the top!
    let dynamic_source = format!("{}\n{}", defines.as_source_code_header(), original_source);

    // 3. Write to a temporary file so Slang can load it natively.
    // We put it in the same folder so that local #includes (like #include "math.slang") still work!
    let temp_module_name = format!("{}_dynamic_compile", file_name);
    let temp_file_path = format!("src/shaders/{}.slang", temp_module_name);
    fs::write(&temp_file_path, &dynamic_source).expect("Failed to write temp shader");

    // Create session and compile the TEMP file
    let mut session = global_session.create_session(&session_desc).unwrap();
    
    // Catch the module loading result
    let module_result = session.load_module(&temp_module_name);

    // 4. CRITICAL: Clean up the temp file immediately so it doesn't pollute your workspace, 
    // even if compilation fails.
    let _ = fs::remove_file(&temp_file_path);

    // Unwrap the result *after* cleaning up the file
    let module = module_result.expect("Failed to load/compile Slang module");

    // 5. Gather all entry points dynamically
    let mut components = vec![module.downcast().clone()];
    for i in 0..module.entry_point_count() {
        components.push(module.entry_point_by_index(i).unwrap().downcast().clone());
    }

    // 6. Link and extract SPIR-V
    let program = session.create_composite_component_type(&components)
        .expect("Failed to create composite type");
    
    let linked = program.link().expect("Failed to link shader");
    let spirv = linked.target_code(0).expect("Failed to generate SPIR-V");

    spirv.as_slice().to_vec()
}