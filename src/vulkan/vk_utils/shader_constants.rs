use std::collections::HashMap;

#[derive(Default)]
pub struct ShaderCompileTimeConstants {
    entries: HashMap<String, String>,
}

impl ShaderCompileTimeConstants {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(mut self, key: &str, value: impl ToString) -> Self {
        self.entries.insert(key.to_string(), value.to_string());
        self
    }

    /// Generates the raw Slang/HLSL code to inject at the top of the file
    pub fn as_source_code_header(&self) -> String {
        let mut header = String::new();
        for (k, v) in &self.entries {
            header.push_str(&format!("#define {} {}\n", k, v));
        }
        header
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, String, String> {
        self.entries.iter()
    }
}