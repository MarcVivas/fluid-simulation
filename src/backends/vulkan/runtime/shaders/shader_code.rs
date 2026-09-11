/// Compiled shader bytes and their diagnostic name. No GPU allocation.
#[derive(Clone, Copy, Debug)]
pub struct ShaderCode {
    pub name: &'static str,
    pub spirv: &'static [u8],
}

/// Embeds a shader built from a path relative to src/backends/vulkan,
/// without the .shader.slang suffix.
#[macro_export]
macro_rules! shader {
    ($path:literal) => {
        $crate::backends::vulkan::runtime::shaders::ShaderCode {
            name: $path,
            spirv: include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $path, ".spv")),
        }
    };
}

/// Checks compiled interfaces without creating a Vulkan device. Called by each
/// owning pass's tests, so there is no central list of shaders to maintain.
#[cfg(test)]
pub(crate) fn assert_interface(shader: ShaderCode, model: u32, entries: &[&str], constants: u32) {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(shader.spirv)).unwrap();
    let mut instructions = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        assert!(
            count > 0 && offset + count <= words.len(),
            "{}",
            shader.name
        );
        instructions.push(&words[offset..offset + count]);
        offset += count;
    }
    let mut found_entries = Vec::new();
    let mut found_ids = Vec::new();
    for i in &instructions {
        match i[0] & 0xffff {
            15 => {
                // OpEntryPoint
                assert_eq!(i[1], model, "{}", shader.name);
                let bytes: Vec<_> = i[3..].iter().flat_map(|w| w.to_le_bytes()).collect();
                let end = bytes.iter().position(|b| *b == 0).unwrap();
                found_entries.push(String::from_utf8(bytes[..end].to_vec()).unwrap());
            }
            71 if i[2] == 1 => {
                // OpDecorate SpecId
                found_ids.push(i[3]);
                let constant = instructions
                    .iter()
                    .find(|c| c[0] & 0xffff == 50 && c[2] == i[1])
                    .expect("Expected scalar specialization constant");
                assert!(
                    instructions.iter().any(|t| t[0] & 0xffff == 21
                        && t[1] == constant[1]
                        && t[2] == 32
                        && t[3] == 0),
                    "{}: specialization values must be u32",
                    shader.name
                );
                if i[3] == 0 {
                    assert!(
                        instructions
                            .iter()
                            .any(|m| m[0] & 0xffff == 331 && m[2] == 38 && m[3] == constant[2]),
                        "{}: first specialization controls local size X",
                        shader.name
                    );
                }
            }
            _ => {}
        }
    }
    found_entries.sort();
    let mut expected_entries = entries.to_vec();
    expected_entries.sort();
    assert_eq!(found_entries, expected_entries, "{}", shader.name);
    found_ids.sort_unstable();
    assert_eq!(
        found_ids,
        (0..constants).collect::<Vec<_>>(),
        "{}",
        shader.name
    );
}
