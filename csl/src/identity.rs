//! Identity hasher for u64

use std::hash::{Hash, Hasher};

#[derive(PartialEq, Eq, Clone, Copy, Debug, PartialOrd, Ord)]
pub(crate) struct Identity64 {
    x: u64,
}

impl Identity64 {
    pub(crate) fn from(x: u64) -> Self {
        Identity64 { x }
    }
}

impl Hash for Identity64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.x)
    }
}

pub(crate) struct IdentityHasher {
    state: u64,
}

impl Default for IdentityHasher {
    fn default() -> Self {
        IdentityHasher { state: 0 }
    }
}

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.state
    }
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!()
    }
    fn write_u8(&mut self, _i: u8) {
        unimplemented!()
    }
    fn write_u16(&mut self, _i: u16) {
        unimplemented!()
    }
    fn write_u32(&mut self, _i: u32) {
        unimplemented!()
    }
    fn write_u64(&mut self, i: u64) {
        self.state = i;
    }
    fn write_usize(&mut self, _i: usize) {
        unimplemented!()
    }
    fn write_i8(&mut self, _i: i8) {
        unimplemented!()
    }
    fn write_i16(&mut self, _i: i16) {
        unimplemented!()
    }
    fn write_i64(&mut self, _i: i64) {
        unimplemented!()
    }
    fn write_i32(&mut self, _i: i32) {
        unimplemented!()
    }
    fn write_isize(&mut self, _i: isize) {
        unimplemented!()
    }
}
