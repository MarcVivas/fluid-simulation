

/// In debug mode, the shaders are compiled at RUNTIME
#[cfg(debug_assertions)]
fn main(){
    
}

// End debug mode


// In release mode, the shaders are compiled at COMPILE TIME
#[cfg(not(debug_assertions))]
const SHADERS_PATH: &str = "src/shaders";

/// The build file compiles all the shaders in the shaders folder. 
/// Executed automatically after compiling the project and before running it.  
#[cfg(not(debug_assertions))]
fn main() {
    use std::{
        env,
        fs::File,
        io::Write,
        path::PathBuf,
    };
    
    use shader_slang::Downcast;
    
    println!("cargo:rerun-if-changed={}", SHADERS_PATH);
    let global_session = shader_slang::GlobalSession::new().unwrap();
    let search_path = std::ffi::CString::new(SHADERS_PATH).unwrap();

    // All compiler options are available through this builder.
    let session_options = shader_slang::CompilerOptions::default()
        //.optimization(shader_slang::OptimizationLevel::High)
        .vulkan_use_entry_point_name(true)
        .matrix_layout_row(true);

    let target_desc = shader_slang::TargetDesc::default().format(shader_slang::CompileTarget::Spirv);

    let targets = [target_desc];
    let search_paths = [search_path.as_ptr()];

    let session_desc = shader_slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let mut session = global_session.create_session(&session_desc).unwrap();

    let mut dir_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    dir_path.push(SHADERS_PATH);
    let dir = std::fs::read_dir(dir_path).unwrap();

    for entry in dir.flatten() {
        let file_name = entry.file_name().into_string().unwrap();
        let file_name = file_name.split(".").next().unwrap();
        load_module(&mut session, file_name);
    }
}


#[cfg(not(debug_assertions))]
fn load_module(session: &mut shader_slang::Session, file_name: &str) {
    let module = session.load_module(&file_name.to_string()).unwrap();

    // 1. Start the list of components with the Module itself
    // We need to use a Vector because we don't know how many entry points exist yet
    let mut components = vec![module.downcast().clone()];

    // 2. Iterate dynamically over all entry points defined in the .slang file
    let entry_point_count = module.entry_point_count();

    // If no entry points are found, we can't compile a program
    if entry_point_count == 0 {
        println!("cargo:warning=No entry points found in {file_name}.slang");
        return;
    }

    for i in 0..entry_point_count {
        let entry_point = module.entry_point_by_index(i).unwrap();
        // Add every entry point (main, count, scan, etc.) to the components list
        components.push(entry_point.downcast().clone());
    }

    // 3. Create the composite program with ALL discovered entry points
    let program = session
        .create_composite_component_type(&components)
        .expect(&format!("Failed to create composite type for {file_name}"));

    let linked_program = program.link().expect(&format!("Failed to link {file_name}"));
    let shader_bytecode = linked_program.target_code(0).expect("Failed to get SPIR-V");

    // ... (The rest of your file writing logic remains the same) ...
    let raw = shader_bytecode.as_slice();
    let length = raw.len();

    let out_dir = env::var("OUT_DIR").unwrap();
    let mut path = PathBuf::from(out_dir);
    path.push(format!("{file_name}.spv"));

    let mut file = File::create(&path).unwrap();
    file.write_all(raw).unwrap();

    let path_str = path.to_str().unwrap();
    println!("cargo:rustc-env={file_name}.spv={path_str}");
    println!("cargo:warning=Compiled {file_name}! {length} bytes, {entry_point_count} entry points.");
}

// End of release mode