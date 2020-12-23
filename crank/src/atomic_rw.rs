//! An atomic-based read-write lockable U32.

use std::sync::atomic::{AtomicU64, Ordering};

/// A read-write lockable U32, backed by an atomic U64.
///
/// Only supports try-locking.
pub struct Rwu32 {
    /// The backing u64 is split up as follows:
    ///
    /// |--- payload (32 bits) ---|W|--- read count (31 bits) ---|
    ///
    /// That is, the higher 32 bits host the actual U32 value,
    /// the 32nd bit is the write-lock bit, and the lower 31 bits are the read
    /// count.
    inner: AtomicU64,
}

const WRITE_BIT_MASK: u64 = 1 << 31;
const READ_SEM_MASK: u64 = (1 << 31) - 1;

impl Rwu32 {
    /// Initialize an `Rwu32` with an initial `u32` value.
    pub fn new(init: u32) -> Self {
        let inner = AtomicU64::new(to_payload(init));
        Self { inner }
    }

    /// Attempt to acquire a write lock.
    pub fn try_write_lock<'a>(&'a self) -> Option<WriteGuard<'a>> {
        let prev = self.inner.fetch_or(WRITE_BIT_MASK, Ordering::Relaxed);
        match State::from(prev) {
            State::RunlockedWlocked | State::RlockedWlocked => {
                // A writer already owns this, so our fetch_or was a no-op.
                // The other writer will unlock the write bit later.
                None
            }
            State::RlockedWunlocked => {
                // We just locked this but have no right to modify due to read lock.
                // Let's remove the write bit.
                self.inner.fetch_and(!WRITE_BIT_MASK, Ordering::Relaxed);
                None
            }
            State::RunlockedWunlocked => {
                // Was unlocked, now locked by us!
                Some(WriteGuard::from(self, from_payload(prev)))
            }
        }
    }

    /// Attempt to acquire a read lock and simultaneously read the current
    /// value. As long as the read lock is held the value is guaranteed not to
    /// change.
    pub fn try_read_lock<'a>(&'a self) -> Option<(u32, ReadGuard<'a>)> {
        let prev = self.inner.fetch_add(1, Ordering::Relaxed);
        match State::from(prev) {
            State::RunlockedWlocked | State::RlockedWlocked => {
                // A writer already owns this, so we should undo our count-up
                // and not look at the value.
                let result = self.inner.fetch_sub(1, Ordering::Relaxed);
                debug_assert!(result & ((1 << 31) - 1) > 0);
                None
            }
            State::RlockedWunlocked | State::RunlockedWunlocked => {
                // We successfully bumped up the count and have thus acquired a
                // read lock. In principle, we should check prev doesn't have 1 << 31 readers.
                Some((from_payload(prev), ReadGuard { rwu32: self }))
            }
        }
    }

    /// Asserts that we are currently unlocked for both read and write, then extracts value.
    pub fn into_inner(self) -> u32 {
        let prev = self.inner.load(Ordering::Relaxed);
        match State::from(prev) {
            State::RunlockedWlocked | State::RlockedWlocked | State::RlockedWunlocked => {
                panic!("Rwu32 was still locked!");
            }
            State::RunlockedWunlocked => from_payload(prev),
        }
    }
}

fn to_payload(v: u32) -> u64 {
    u64::from(v) << 32
}

fn from_payload(v: u64) -> u32 {
    (v >> 32) as u32
}

enum State {
    RlockedWlocked,
    RunlockedWlocked,
    RlockedWunlocked,
    RunlockedWunlocked,
}

impl State {
    fn from(inner: u64) -> Self {
        match (inner & READ_SEM_MASK > 0, inner & WRITE_BIT_MASK > 0) {
            (true, true) => Self::RlockedWlocked,
            (false, true) => Self::RunlockedWlocked,
            (true, false) => Self::RlockedWunlocked,
            (false, false) => Self::RunlockedWunlocked,
        }
    }
}

pub struct WriteGuard<'a> {
    rwu32: &'a Rwu32,
    previous: u32,
    current: u32,
}

impl<'a> WriteGuard<'a> {
    fn from(rwu32: &'a Rwu32, value: u32) -> Self {
        Self {
            rwu32,
            previous: value,
            current: value,
        }
    }

    pub fn write(&mut self, v: u32) {
        self.current = v
    }
}

impl<'a> Drop for WriteGuard<'a> {
    fn drop(&mut self) {
        if self.current > self.previous {
            // use a single atomic add which is 1 << 31 below of updating
            // the value correctly, to be cancelled out by the write bit.
            let mut diff = to_payload(self.current - self.previous);
            debug_assert!(diff > (1 << 31));
            diff -= 1 << 31;
            let result = self.rwu32.inner.fetch_add(diff, Ordering::Relaxed);
            debug_assert!(result & WRITE_BIT_MASK == 1);
            debug_assert!(from_payload(result) == self.previous);
        } else {
            // similarly, delete the extra write lock bit
            let mut diff = to_payload(self.previous - self.current);
            diff += 1 << 31;
            let result = self.rwu32.inner.fetch_sub(diff, Ordering::Relaxed);
            debug_assert!(result & WRITE_BIT_MASK == 1);
            debug_assert!(from_payload(result) == self.previous);
        }
    }
}

pub struct ReadGuard<'a> {
    rwu32: &'a Rwu32,
}

impl<'a> Drop for ReadGuard<'a> {
    fn drop(&mut self) {
        let result = self.rwu32.inner.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(result & ((1 << 31) - 1) > 0);
    }
}
