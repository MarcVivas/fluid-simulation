use std::{fs, path::{Path, PathBuf}};


const SLANG_VERSION: &str = "2026.11"; 

// Downloads slang sdk 
fn main(){
    let project_root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let sdk_dir = project_root.join("target/slang-sdk");
    
    let bin_subpath = match std::env::consts::OS {
            "windows" => "bin/windows-x64/release/slangc.exe",
            "linux" => "bin/linux-x64/release/slangc",
            "macos" => "bin/macos-x64/release/slangc",
            _ => panic!("Unsupported OS"),
        };

    let slangc_path = sdk_dir.join(bin_subpath);
    
    if !slangc_path.exists() {
        if sdk_dir.exists() {
            println!("Detected corrupt or incomplete Slang SDK. Cleaning up...");
            let _ = fs::remove_dir_all(&sdk_dir);
        }
            
        download_and_extract_slang(&sdk_dir);
    }

    // Tell Cargo to rerun only if build.rs itself changes
    println!("cargo:rerun-if-changed=build.rs");
    
}

fn download_and_extract_slang(download_dir: &Path){
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let url = get_slang_download_url(os, arch);

    // Create download directory if not present 
    fs::create_dir_all(download_dir).expect("Failed to create download directory");

    // Download slang
    println!("Downloading Slang from: {}", url); 

    let response = ureq::get(&url)
        .call()
        .expect("Failed to download Slang; check your internet connection");
    
    let reader = response.into_reader();

    // Extract the file 
    let tar_gz = flate2::read::GzDecoder::new(reader);
    let mut archive = tar::Archive::new(tar_gz);

    archive
        .unpack(download_dir)
        .expect("Failed to extract tar.gz archive");

    println!("Slang successfuly downloaded and extracted to: {:?}", download_dir);    
    
}

fn get_slang_download_url(os: &str, arch: &str) -> String {
    match os {
        "windows" => format!(
            "https://github.com/shader-slang/slang/releases/download/v{0}/slang-{0}-windows-x86_64.tar.gz",
            SLANG_VERSION
        ),
        
        "linux" => format!(
            "https://github.com/shader-slang/slang/releases/download/v{0}/slang-{0}-linux-x86_64.tar.gz",
            SLANG_VERSION
        ),

        "macos" => {
            let mac_arch = if arch == "aarch64" {"arm64"} else {"x86_64"};
            format!(
                "https://github.com/shader-slang/slang/releases/download/v{0}/slang-{0}-macos-{1}.tar.gz",
                SLANG_VERSION, mac_arch
            )
        },
        _ => panic!("Unsupported operating system: {}", os)
    }
}