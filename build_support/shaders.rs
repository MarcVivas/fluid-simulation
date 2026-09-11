use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub fn find_compiler(dir: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    };
    let mut paths: Vec<_> = fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    paths.sort();
    for path in paths {
        if path.is_file() && path.file_name()?.to_str()? == name {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_compiler(&path) {
                return Some(found);
            }
        }
    }
    None
}

fn shader_directories(dir: &Path, directories: &mut Vec<PathBuf>, shaders: &mut Vec<PathBuf>) {
    // Directory watches also detect newly added or removed imports.
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut paths: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    if paths
        .iter()
        .any(|p| p.extension().is_some_and(|e| e == "slang"))
    {
        directories.push(dir.to_owned());
    }
    for path in paths {
        if path.is_dir() {
            shader_directories(&path, directories, shaders);
        } else if path.to_string_lossy().ends_with(".shader.slang") {
            shaders.push(path);
        }
    }
}

pub fn compile(root: &Path, compiler: &Path) {
    println!("cargo:rerun-if-changed={}", compiler.display());
    let mut directories = Vec::new();
    let source_root = root.join("src/backends/vulkan");
    let mut shaders = Vec::new();
    shader_directories(&source_root, &mut directories, &mut shaders);
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    // Validate during builds when SPIR-V Tools are installed. An explicit override
    // is required to work, so CI can make validation mandatory.
    println!("cargo:rerun-if-env-changed=SPIRV_VAL");
    let validator_override = std::env::var_os("SPIRV_VAL");
    let validator = validator_override
        .clone()
        .unwrap_or_else(|| "spirv-val".into());
    let validate = Command::new(&validator)
        .arg("--version")
        .output()
        .is_ok_and(|result| result.status.success());
    assert!(
        validator_override.is_none() || validate,
        "SPIRV_VAL must name a working spirv-val executable"
    );
    if !validate {
        println!(
            "cargo:warning=spirv-val not found; shaders will compile without SPIR-V validation"
        );
    }
    // Remove only this build script's generated shader subtree. Otherwise a
    // deleted source could still satisfy include_bytes! using stale SPIR-V.
    let shader_output_dir = output_dir.join("shaders");
    if shader_output_dir.exists() {
        fs::remove_dir_all(&shader_output_dir).expect("Cannot clear generated shaders");
    }
    for source in shaders {
        let relative = source.strip_prefix(&source_root).unwrap();
        let output = shader_output_dir
            .join(relative)
            .with_extension("")
            .with_extension("spv");
        fs::create_dir_all(output.parent().unwrap())
            .expect("Cannot create shader output directory");
        let mut command = Command::new(compiler);
        command.arg(&source).args([
            "-target",
            "spirv",
            "-profile",
            "spirv_1_5",
            "-fvk-use-entrypoint-name",
            "-matrix-layout-row-major",
        ]);
        for dir in &directories {
            command.arg("-I").arg(dir);
        }
        let result = command
            .arg("-o")
            .arg(&output)
            .output()
            .expect("Failed to execute Slang compiler");
        assert!(
            result.status.success(),
            "Slang compilation failed for {}:\n{}\n{}",
            source.display(),
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        if validate {
            let result = Command::new(&validator)
                .args(["--target-env", "vulkan1.3", "--scalar-block-layout"])
                .arg(&output)
                .output()
                .expect("Failed to run spirv-val");
            assert!(
                result.status.success(),
                "SPIR-V validation failed for {}:\n{}",
                source.display(),
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}
