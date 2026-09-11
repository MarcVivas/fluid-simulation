use std::collections::HashMap;

pub struct GpuProfilingZones {
    zone_map: HashMap<String, u32>,
    next_index: u32,
    max_queries: u32,
}

impl GpuProfilingZones {
    pub fn new(max_queries: u32) -> Self {
        Self {
            zone_map: HashMap::new(),
            max_queries,
            next_index: 0,
        }
    }

    pub fn get_or_create_zone_index(&mut self, label: &str) -> u32 {
        if let Some(&index) = self.zone_map.get(label) {
            return index;
        }
        let index = self.next_index;
        assert!(
            index + 1 < self.max_queries,
            "Exceeded max GPU profiler zones"
        );
        self.zone_map.insert(label.to_string(), index);
        self.next_index += 2;
        index
    }

    #[allow(unused)]
    pub fn get_zone_index(&self, label: &str) -> Option<&u32> {
        self.zone_map.get(label)
    }

    pub fn all_zones(&self) -> &HashMap<String, u32> {
        &self.zone_map
    }

    pub fn get_active_query_count(&self) -> u32 {
        self.next_index
    }
}
