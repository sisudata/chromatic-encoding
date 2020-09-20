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
use pdatastructs::{filters::bloomfilter::BloomFilter, filters::Filter, hyperloglog::HyperLogLog};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::hash::{Hash, Hasher};
use std::iter::repeat_with;
use std::slice;
use std::sync::Mutex;

pub(crate) type Vertex = u32;

pub struct RollingBloom {
    blooms: Vec<BloomFilter<Edge, BuildHasherDefault<ThreadUnsafeHasher>>>,
}

/// Compressed representation of two vertices through bit concatenation.
#[derive(Default, PartialEq, Hash, PartialOrd, Eq, Ord, Clone)]
pub(crate) struct Edge {
    e: u64,
}

#[derive(Default, Clone)]
pub(crate) struct GraphStats {
    nlines: usize,
    nskip: usize,
    max_nnz: usize,
    sum_nnz: usize,
    sum_edges: u128,
}

impl GraphStats {
    fn update(&mut self, nnz: usize, nskip: usize) {
        self.nskip += nskip;
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
        self.nskip += other.nskip;
    }

    pub(crate) fn print(&self) {
        let avg_nnz = self.sum_nnz / self.nlines;
        let avg_edges = self.sum_edges / self.nlines as u128;
        println!("num edges filtered by bloom {}", self.nskip);
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
    cms: &RollingBloom,
    threshold_k: u8,
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
            || {
                EdgeCollector::new(
                    featurizer,
                    cms,
                    writers,
                    shared_stats,
                    nthreads,
                    threshold_k,
                )
            },
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
    cms: &'a RollingBloom,
    threshold_k: u8,
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
        cms: &'b RollingBloom,
        writers: &'b Vec<Mutex<EdgeMap>>,
        shared_stats: &'b Mutex<GraphStats>,
        nthreads: usize,
        threshold_k: u8,
    ) -> EdgeCollector<'b> {
        EdgeCollector {
            featurizer,
            cms,
            writers,
            locals: vec![Default::default(); writers.len()],
            is_ready: vec![false; nthreads],
            indices: Vec::with_capacity(featurizer.nsparse()),
            repeats: vec![false; featurizer.nsparse()],
            stats: Default::default(),
            shared_stats,
            threshold_k,
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
        let mut nskipped = 0usize;
        for (i, j) in self.indices.iter().copied().tuple_combinations() {
            let e = Edge::new(i, j);
            if self.cms.query(&e) < self.threshold_k as usize {
                nskipped += 1;
                continue;
            }
            Self::save_local(&mut self.locals, e);
        }

        let nnz = self.indices.len();
        self.stats.update(nnz, nskipped);

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

struct EdgeVisitor<'a, R> {
    featurizer: &'a Featurizer,
    indices: Vec<Vertex>,
    repeats: Vec<bool>,
    reduced: R,
}

impl<'a, R> EdgeVisitor<'a, R> {
    fn new<'b>(featurizer: &'b Featurizer, r: R) -> EdgeVisitor<'b, R> {
        EdgeVisitor::<'b, R> {
            featurizer,
            indices: Vec::with_capacity(featurizer.nsparse()),
            repeats: vec![false; featurizer.nsparse()],
            reduced: r,
        }
    }

    fn consume_line<'b, F>(&mut self, word_iter: DelimIter<'b>, reduce: F)
    where
        F: Fn(&mut R, Edge),
    {
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
            reduce(&mut self.reduced, Edge::new(i, j));
        }

        for &idx in &self.indices {
            self.repeats[idx as usize] = false;
        }
        self.indices.clear();
    }

    fn merge<F>(mut self, other: Self, merge: F) -> Self
    where
        F: Fn(&mut R, R),
    {
        merge(&mut self.reduced, other.reduced);
        self
    }
}

struct CountAndHLL {
    count: usize,
    hll: HyperLogLog<Edge, BuildHasherDefault<ThreadUnsafeHasher>>,
}

/// returns total edge count and unique edge count
pub(crate) fn estimate_edges(train: &SvmScanner, featurizer: &Featurizer) -> (usize, usize) {
    let candh = {
        train
            .fold_reduce(
                || {
                    EdgeVisitor::<CountAndHLL>::new(
                        featurizer,
                        CountAndHLL {
                            count: 0,
                            hll: HyperLogLog::<_, _>::with_hash(
                                18,
                                BuildHasherDefault::<ThreadUnsafeHasher>::default(),
                            ),
                        },
                    )
                },
                |mut x, line| {
                    x.consume_line(line, |candh, edge| {
                        candh.count += 1;
                        candh.hll.add(&edge);
                    });
                    x
                },
                |x, y| {
                    x.merge(y, |l: &mut CountAndHLL, r: CountAndHLL| {
                        l.count += r.count;
                        l.hll.merge(&r.hll);
                    })
                },
            )
            .reduced
    };
    (candh.count, candh.hll.count())
}

impl RollingBloom {
    fn new(k: usize, bits: usize, hashes: usize) -> Self {
        let mut v = Vec::new();
        for i in 0..k {
            v.push(BloomFilter::with_params_and_hash(
                bits,
                hashes,
                BuildHasherDefault::<ThreadUnsafeHasher>::default(),
            ))
        }
        RollingBloom { blooms: v }
    }

    /// occurs at least (num returns) times
    fn query(&self, e: &Edge) -> usize {
        self.blooms
            .iter()
            .position(|bloom| !bloom.query(e))
            .unwrap_or(self.blooms.len())
    }

    fn push(&mut self, e: &Edge, k: usize) {
        self.blooms[k].insert(e).unwrap();
    }

    fn merge(&mut self, o: &Self) {
        assert!(self.blooms.len() == o.blooms.len());
        for i in 0..self.blooms.len() {
            self.blooms[i].union(&o.blooms[i]);
        }
    }
}

/// creates bloom filter
pub(crate) fn create_cms(
    train: &SvmScanner,
    featurizer: &Featurizer,
    threshold_k: u8,
    n_edges: usize,
    n_unique_edges: usize,
) -> RollingBloom {
    let max_per_thread = 1. * 1024. * 1024. * 1024.;
    let max_bits_per_bloom = (max_per_thread * 8. / (threshold_k as f64)) as usize;
    println!(
        "at {}G max per thread, {} rolling blooms",
        max_per_thread / 1024. / 1024. / 1024.,
        threshold_k
    );
    let num_hashes =
        ((max_bits_per_bloom as f64 * (2f64).ln() / n_unique_edges as f64) as usize).max(1);
    println!(
        "{}MB per bloom, {} hashes",
        max_bits_per_bloom / 1024 / 1024,
        num_hashes
    );
    let cms = {
        train
            .fold_reduce(
                || {
                    EdgeVisitor::new(
                        featurizer,
                        RollingBloom::new(threshold_k as usize, max_bits_per_bloom, num_hashes),
                    )
                },
                |mut x, line| {
                    x.consume_line(line, |cms, edge| {
                        let occurence = cms.query(&edge);
                        if occurence < threshold_k as usize {
                            cms.push(&edge, occurence);
                        }
                    });
                    x
                },
                |x, y| {
                    x.merge(y, |l: &mut RollingBloom, r| {
                        l.merge(&r);
                    })
                },
            )
            .reduced
    };
    cms

    // TODO: rolling bloom instead, should be more reasonable.
    // scaling blooms?x
}
