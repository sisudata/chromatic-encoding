//! Compact graph data structure.

use itertools::EitherOrBoth::{Both, Right};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::convert::TryInto;
use std::iter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

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
    /// from `i`, not necessarily sorted.
    pub(crate) fn new(offsets: Vec<usize>, neighbors: Vec<Vertex>) -> Self {
        assert!(offsets.len() <= (1 << 32));
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

    pub(crate) fn nvertices(&self) -> usize {
        self.offsets.len() - 1
    }
}
