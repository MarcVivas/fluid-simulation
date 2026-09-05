pub struct FramePacer {
    global_frame_counter: u64,
    frames_in_flight: usize, 
}

impl FramePacer {

    pub fn new(frames_in_flight: usize) -> Self {
        Self { global_frame_counter: 0, frames_in_flight }
    }

    /// Advance to the next frame
    pub fn advance(&mut self){
        self.global_frame_counter += 1;
    }

    /// Get the total number of frames in flight. Frame latency
    pub fn frames_in_flight(&self) -> usize {
        self.frames_in_flight
    }

    /// Returns the index for flame in flight ring buffers
    pub fn ring_index(&self) -> usize {
        self.global_frame_counter as usize % self.frames_in_flight 
    }

    pub fn total_frames_processed(&self) -> u64 {
        self.global_frame_counter
    }
}