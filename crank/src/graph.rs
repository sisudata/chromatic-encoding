//! Compact graph data structure.

use itertools::EitherOrBoth::{Both, Right};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::convert::TryInto;
use std::iter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::iter::IndexedParallelIterator;
use rayon::slice::ParallelSlice;

pub(crate) type Vertex = u32;

/// A compact adjacency list intended for sparse graphs.
///
/// The space of vertices is a contiguous range of u32 ints
/// from [0, nvertices).
pub struct Graph {
    offsets: Vec<usize>,
    neighbors: Vec<Vertex>,
}

impl Graph {
    /// `offsets.len()` should be one greater than the number of vertices
    /// with `neighbors[offsets[i]..offsets[i+1]]` being the edges incident
    /// from `i`, which should be necessarily sorted and bidirectional.
    pub(crate) fn new(offsets: Vec<usize>, neighbors: Vec<Vertex>) -> Self {
        assert!(offsets.len() <= (1 << 32));
        debug_assert!(offsets.par_windows(2).enumerate().all(|(i, s)| {
            s[0] < s[1]
                && neighbors[s[0]..s[1]].windows(2).all(|ss| ss[0] < ss[1])
                && neighbors[s[0]..s[1]].iter().copied().all(|j| {
                    let ref i = i as u32;
                    neighbors[offsets[j as usize]..offsets[1 + j as usize]]
                        .binary_search(i)
                        .is_ok()
                })
        }));
        Self { offsets, neighbors }
    }

    pub(crate) fn neighbors(&self, v: Vertex) -> &[Vertex] {
        let v = v as usize;
        let lo = self.offsets[v];
        let hi = self.offsets[v + 1];
        &self.neighbors[lo..hi]
    }

    pub(crate) fn degree(&self, v: Vertex) -> usize {
        let v = v as usize;
        let lo = self.offsets[v];
        let hi = self.offsets[v + 1];
        hi - lo
    }

    pub fn nvertices(&self) -> usize {
        self.offsets.len() - 1
    }
}
