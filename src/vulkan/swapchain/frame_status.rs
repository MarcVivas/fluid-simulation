pub enum FrameStatus {
    Ready { image_index: u32 },
    OutOfDate,
}