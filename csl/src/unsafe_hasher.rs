//! This module provides an unsafe send/sync implementation
//! wrapper to fasthash::xx hasher.
//!
//! Callers should take care to only create and destroy such hashers
//! in a single thread, which keeps the internal pointer that the hasher
//! relies on valid for the duration of sharing.

use fasthash::XXHasher;
use std::hash::{Hash, Hasher};

#[derive(Default)]
pub struct ThreadUnsafeHasher {
    hasher: XXHasher,
}

/// See module documentation. Callers must pinky-promise to not abuse this.
unsafe impl Send for ThreadUnsafeHasher {}
unsafe impl Sync for ThreadUnsafeHasher {}

impl Hasher for ThreadUnsafeHasher {
    fn finish(&self) -> u64 {
        self.hasher.finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        bytes.hash(&mut self.hasher)
    }
    fn write_u8(&mut self, i: u8) {
        i.hash(&mut self.hasher)
    }
    fn write_u16(&mut self, i: u16) {
        i.hash(&mut self.hasher)
    }
    fn write_u32(&mut self, i: u32) {
        i.hash(&mut self.hasher)
    }
    fn write_u64(&mut self, i: u64) {
        i.hash(&mut self.hasher);
    }
    fn write_usize(&mut self, i: usize) {
        i.hash(&mut self.hasher)
    }
    fn write_i8(&mut self, i: i8) {
        i.hash(&mut self.hasher)
    }
    fn write_i16(&mut self, i: i16) {
        i.hash(&mut self.hasher)
    }
    fn write_i64(&mut self, i: i64) {
        i.hash(&mut self.hasher)
    }
    fn write_i32(&mut self, i: i32) {
        i.hash(&mut self.hasher)
    }
    fn write_isize(&mut self, i: isize) {
        i.hash(&mut self.hasher)
    }
}
