pub struct PingPong<T> {
    use_first: bool,
    data: [T; 2],
}

impl <T> PingPong<T> {
    pub fn new(ping: T, pong: T) -> Self {
        Self { use_first: true, data: [ping, pong] }
    }

    pub fn swap(&mut self) {
        self.use_first = !self.use_first;
    }


    /// Helper to get the index of the current buffer
    pub fn current_index(&self) -> usize {
        if self.use_first { 0 } else { 1 }
    }

    fn next_index(&self) -> usize {
        if self.use_first { 1 } else { 0 }
    }

    pub fn current(&self) -> &T {
        &self.data[self.current_index()]
    }

    pub fn next(&self) -> &T {
        &self.data[self.next_index()]
    }

    pub fn read_write(&self) -> (&T, &T) {
        (self.current(), self.next())
    }

    pub fn from_index(&self, index: usize) -> &T {
        debug_assert!(index < self.data.len());
        &self.data[index]
    }
}
