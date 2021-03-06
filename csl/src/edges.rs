//! Graph construction with edge hashing.

// POTENTIAL OPTIMIZATION
//
// Current hashing-based approach is pretty legit, but a representation that stores
// edges in a sorted manner could end up being more compact and cache-friendly.
//
// We basically just want a u64 set, and something like a Judy array could in principle
// do better. If the destination sets were https://github.com/adevore/rudy then that
// could be a 2x based on some simple microbenchmarks in the repo.
//
// Jon Gjengset also thinks that we can improve on the current approach (which focuses
// on read-locality by locking destination tables) by switching to channels that dump
// local hashsets to a destination writers. This'd give better write locality, which
// is arguably more important.

use crate::feature::Featurizer;
use crate::svm_scanner::{DelimIter, SvmScanner};
use crate::unsafe_hasher::ThreadUnsafeHasher;
use hashbrown::HashMap;
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::hash::{Hash, Hasher};
use std::iter::repeat_with;
use std::slice;
use std::sync::Mutex;

pub(crate) type Vertex = u32;

/// Compressed representation of two vertices through bit concatenation.
#[derive(Default, PartialEq, Hash, PartialOrd, Eq, Ord, Clone)]
pub(crate) struct Edge {
    e: u64,
}

#[derive(Default, Clone)]
pub(crate) struct GraphStats {
    nlines: usize,
    max_nnz: usize,
    sum_nnz: usize,
    sum_edges: u128,
}

impl GraphStats {
    fn update(&mut self, nnz: usize) {
        self.nlines += 1;
        self.max_nnz = self.max_nnz.max(nnz);
        self.sum_nnz += nnz;
        self.sum_edges += (nnz * (nnz - 1) / 2) as u128;
    }

    fn merge(&mut self, other: &Self) {
        self.nlines += other.nlines;
        self.max_nnz = self.max_nnz.max(other.max_nnz);
        self.sum_nnz += other.sum_nnz;
        self.sum_edges += other.sum_edges;
    }

    pub(crate) fn print(&self) {
        let avg_nnz = self.sum_nnz / self.nlines;
        let avg_edges = self.sum_edges / self.nlines as u128;
        println!("avg nnz per row {}", avg_nnz);
        println!("max nnz per row {}", self.max_nnz);
        println!("avg edges per row {}", avg_edges);
    }

    pub(crate) fn max_nnz(&self) -> usize {
        self.max_nnz
    }

    pub(crate) fn nlines(&self) -> usize {
        self.nlines
    }
}

impl Edge {
    /// Requires `x < y`
    pub(crate) fn new(x: Vertex, y: Vertex) -> Edge {
        assert!(x < y);
        Edge {
            e: ((x as u64) << 32) | y as u64,
        }
    }

    pub(crate) fn left(&self) -> Vertex {
        (self.e >> 32) as Vertex
    }

    pub(crate) fn right(&self) -> Vertex {
        (self.e & ((1 << 32) - 1)) as Vertex
    }
}

#[derive(Clone)]
struct EdgeStats {
    memoized_hash: u64,
    count: usize,
}

impl EdgeStats {
    fn new(hash: u64) -> Self {
        EdgeStats {
            memoized_hash: hash,
            count: 0,
        }
    }

    fn add(&mut self, observations: usize) {
        self.count += observations;
    }
}

type EdgeMap = HashMap<Edge, EdgeStats, BuildHasherDefault<ThreadUnsafeHasher>>;

/// Return edge set, frequency of edges, and stats on features active across row (recall
/// features are feature hashes).
pub(crate) fn collect_edges(
    train: &SvmScanner,
    featurizer: &Featurizer,
) -> (Vec<Edge>, Vec<usize>, GraphStats) {
    let nthreads = train.nthreads();
    let ref writers: Vec<Mutex<EdgeMap>> = repeat_with(|| {
        Mutex::new(HashMap::with_hasher(
            BuildHasherDefault::<ThreadUnsafeHasher>::default(),
        ))
    })
    .take(nthreads)
    .collect();
    let ref shared_stats = Mutex::new(GraphStats::default());

    {
        train.for_each(
            || EdgeCollector::new(featurizer, writers, shared_stats, nthreads),
            EdgeCollector::consume_line,
            EdgeCollector::finish,
        );
    }

    let lens: Vec<usize> = writers
        .iter()
        .map(|mutex| {
            let writer = mutex.lock().unwrap();
            writer.len()
        })
        .collect();
    let mut edges = vec![Edge::default(); lens.iter().sum::<usize>()];
    let mut frequencies = vec![0usize; edges.len()];
    {
        let ptr = edges.as_mut_ptr();
        let fptr = frequencies.as_mut_ptr();

        let edge_slices: Vec<_> = lens
            .into_iter()
            .scan(0, |state, size| {
                let start = *state;
                let stop = start + size;
                *state = stop;
                // SAFETY: each slice refers to a disjoint region in the vec
                Some((
                    unsafe { slice::from_raw_parts_mut(ptr.add(start), size) },
                    unsafe { slice::from_raw_parts_mut(fptr.add(start), size) },
                ))
            })
            .collect();

        edge_slices
            .into_par_iter()
            // note this zipped iter is of shorter length (intentionally)
            .zip(writers.into_par_iter())
            .for_each(|((edge_slice, freq_slice), writer)| {
                let writer = writer.lock().unwrap();
                for (i, (e, s)) in writer.iter().enumerate() {
                    edge_slice[i] = e.clone();
                    freq_slice[i] = s.count;
                }
            });

        if cfg!(debug_assertions) {
            let in_vec: HashSet<Edge> = edges.iter().cloned().collect();
            for w in writers {
                let orig_set = w.lock().unwrap();
                for e in orig_set.keys() {
                    debug_assert!(in_vec.contains(e), "{} {}", e.left(), e.right());
                }
            }
        }
    }
    let unlocked_stats = shared_stats.lock().unwrap();
    (edges, frequencies, unlocked_stats.clone())
}

struct EdgeCollector<'a> {
    featurizer: &'a Featurizer,
    writers: &'a Vec<Mutex<EdgeMap>>,
    locals: Vec<EdgeMap>,
    // which writers that are we ready to sync to
    is_ready: Vec<bool>,

    // local per-line processing state
    indices: Vec<Vertex>, // active vertices in line
    repeats: Vec<bool>,   // characteristic vector for above int list

    // stats to gather
    stats: GraphStats,

    // stats to sync
    shared_stats: &'a Mutex<GraphStats>,
}

impl<'a> EdgeCollector<'a> {
    fn new<'b>(
        featurizer: &'b Featurizer,
        writers: &'b Vec<Mutex<EdgeMap>>,
        shared_stats: &'b Mutex<GraphStats>,
        nthreads: usize,
    ) -> EdgeCollector<'b> {
        EdgeCollector {
            featurizer,
            writers,
            locals: vec![Default::default(); writers.len()],
            is_ready: vec![false; nthreads],
            indices: Vec::with_capacity(featurizer.nsparse()),
            repeats: vec![false; featurizer.nsparse()],
            stats: Default::default(),
            shared_stats,
        }
    }

    fn consume_line<'b>(&mut self, word_iter: DelimIter<'b>) {
        // skip the target word

        for (i, word) in word_iter.skip(1).enumerate() {
            let h = match self.featurizer.sparse(i, word) {
                Some(h) => h,
                None => continue,
            };
            if self.repeats[h as usize] {
                continue;
            }
            self.repeats[h as usize] = true;
            self.indices.push(h as Vertex);
        }

        self.indices.sort_unstable();
        for (i, j) in self.indices.iter().copied().tuple_combinations() {
            Self::save_local(&mut self.locals, Edge::new(i, j));
        }

        let nnz = self.indices.len();
        self.stats.update(nnz);

        for &idx in &self.indices {
            self.repeats[idx as usize] = false;
        }
        self.indices.clear();

        self.add_ready();
        self.sync_ready();
    }

    fn save_local(locals: &mut [EdgeMap], e: Edge) {
        let mut s: ThreadUnsafeHasher = Default::default();
        e.hash(&mut s);
        let h = s.finish();
        let w = (h as usize) % locals.len();
        let mut s = EdgeStats::new(h);
        s.add(1);
        Self::add_to_map(&e, &s, &mut locals[w]);
    }

    fn add_to_map(e: &Edge, s: &EdgeStats, m: &mut EdgeMap) {
        m.raw_entry_mut()
            .from_hash(s.memoized_hash, |other| other == e)
            .or_insert_with(|| (e.clone(), EdgeStats::new(s.memoized_hash)))
            .1
            .add(s.count);
    }

    const BUFSIZE: usize = 8 * 1024;

    fn add_ready(&mut self) {
        // more writers = more contention but b/c we're sharding
        // at the same level of parallelism as our writers the expected
        // contention rate is the same
        for i in 0..self.is_ready.len() {
            let curr_len = self.locals[i].len();
            self.is_ready[i] = curr_len >= Self::BUFSIZE
        }
    }

    fn sync_ready(&mut self) {
        for (i, ready) in self.is_ready.iter_mut().enumerate() {
            if !*ready {
                continue;
            }

            if let Ok(mut writer) = self.writers[i].try_lock() {
                for (k, v) in &self.locals[i] {
                    Self::add_to_map(k, v, &mut writer);
                    *ready = false;
                }
            }

            if !*ready {
                self.locals[i].clear();
            }
        }
    }

    fn finish(&mut self) {
        for i in 0..self.writers.len() {
            {
                let mut writer = self.writers[i].lock().unwrap();
                for (k, v) in &self.locals[i] {
                    Self::add_to_map(k, v, &mut writer);
                }
            }
            self.locals[i].clear();
        }
        let mut stats = self.shared_stats.lock().unwrap();
        stats.merge(&self.stats);
    }
}
