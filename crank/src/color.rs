//! The core coloring functionality for sparse datasets represented
//! as pages of sparse matrices.

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::mem;
use std::sync::atomic;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use indicatif::{MultiProgress, ProgressBar, ProgressIterator, ProgressStyle};
use rand_pcg::Lcg64Xsh32;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::{
    atomic_rw::{ReadGuard, Rwu32},
    graph::Graph,
    graph::Vertex,
    SparseMatrix,
};

/// Given the training set, a color mapping, and the number of colors,
/// "remaps" a dataset, generating a vector `remap` such that `remap[f]`
/// is `f`s rank among all features it shares a color with, 1-indexed.
///
/// I.e., the lowest-numbered feature for a given color will have a `remap`
/// of 1, the second lowest numbered, 2, and so on.
pub fn remap(ncolors: u32, colors: &[u32]) -> Vec<u32> {
    let mut color_counts = vec![0u32; ncolors as usize];
    let mut remap = vec![0u32; colors.len()];
    colors.iter().copied().enumerate().for_each(|(f, c)| {
        color_counts[c as usize] += 1;
        remap[f as usize] = color_counts[c as usize]
    });

    remap
}

/// Returns `(ncolors, colors, log_info)` for a max-degree-ordered coloring of the graph.
pub fn greedy(graph: &Graph) -> (u32, Vec<u32>, HashMap<String, f64>) {
    let nvertices = graph.nvertices();
    let mut vertices: Vec<_> = (0..nvertices).map(|v| v as Vertex).collect();
    vertices.sort_unstable_by_key(|&v| graph.degree(v));

    const NO_COLOR: u32 = std::u32::MAX;
    let mut colors: Vec<u32> = vec![NO_COLOR; nvertices];
    let mut adjacent_colors: Vec<bool> = Vec::new();

    for vertex in vertices.into_iter().rev() {
        // loop invariant is that none of adjacent_colors elements are true

        // what color are our neighbors?
        let mut nadjacent_colors = 0;
        for &n in graph.neighbors(vertex) {
            let n = n as usize;
            if colors[n] == NO_COLOR {
                continue;
            }

            let c = colors[n] as usize;
            if !adjacent_colors[c] {
                adjacent_colors[c] = true;
                nadjacent_colors += 1;
                if nadjacent_colors == colors.len() {
                    break;
                }
            }
        }

        // what's the smallest color not in our neighbors?
        let chosen = if nadjacent_colors == adjacent_colors.len() {
            adjacent_colors.push(false);
            adjacent_colors.len() - 1
        } else {
            adjacent_colors.iter().copied().position(|x| !x).unwrap()
        };
        colors[vertex as usize] = chosen as u32;

        // retain loop invariant, unset neighbor colors
        if graph.degree(vertex) >= adjacent_colors.len() {
            graph
                .neighbors(vertex)
                .iter()
                .map(|&n| colors[n as usize])
                .filter(|&n| n != NO_COLOR)
                .for_each(|c| {
                    adjacent_colors[c as usize] = false;
                });
        } else {
            for c in adjacent_colors.iter_mut() {
                *c = false;
            }
        }
    }

    let ncolors = adjacent_colors.len();
    (
        ncolors as u32,
        colors,
        [("greedy_ncolors".to_owned(), ncolors as f64)]
            .iter()
            .cloned()
            .collect(),
    )
}

/// Returns `(ncolors, colors)` for a Glauber-colored graph. If greedy colors
/// requires more colors than specified, then increases the color count to that many.
pub fn glauber(
    graph: &Graph,
    ncolors: u32,
    nsamples: usize,
) -> (u32, Vec<u32>, HashMap<String, f64>) {
    let greedy_start = Instant::now();
    let (greedy_ncolors, colors, _) = greedy(graph);
    let greedy_time = Instant::now().duration_since(greedy_start);
    assert!(
        greedy_ncolors <= ncolors,
        "greedy ncolors {} budget {}",
        greedy_ncolors,
        ncolors
    );

    // https://www.math.cmu.edu/~af1p/Texfiles/colorbdd.pdf
    // https://www.math.cmu.edu/~af1p/Teaching/MCC17/Papers/colorJ.pdf
    // run glauber markov chain on a coloring
    // chain sampling can be parallel with some simple conflict detection

    let colors = colors
        .into_iter()
        .map(|x| Rwu32::new(x))
        .collect::<Vec<_>>();
    let nthreads = rayon::current_num_threads() as usize;

    let glauber_start = Instant::now();
    let conflicts = (0..nthreads)
        .into_par_iter()
        .map(|i| {
            let nsamples = (nsamples + nthreads - 1) / nthreads;
            let mut rng = Lcg64Xsh32::new(0xcafef00dd15ea5e5, i as u64);
            let mut conflicts = 0;
            let mut viable_colors = DiscreteSampler::new(ncolors);
            let mut neighbor_guards = Vec::new();

            for _ in 0..nsamples {
                loop {
                    let successful = try_mcmc_update(
                        &mut rng,
                        &colors,
                        &graph,
                        &mut viable_colors,
                        &mut neighbor_guards,
                    );
                    neighbor_guards.clear();
                    if successful.is_some() {
                        break;
                    }
                    conflicts += 1;
                }
            }
            conflicts
        })
        .sum::<usize>();
    let glauber_time = Instant::now().duration_since(glauber_start);

    let colors = colors.into_iter().map(|x| x.into_inner()).collect();

    // implement parallel monte carlo with *SORTED* neighbors
    // make neighbors atomic u64s
    (
        ncolors,
        colors,
        [
            ("greedy_ncolors".to_owned(), greedy_ncolors as f64),
            ("nsamples".to_owned(), nsamples as f64),
            ("conflicts".to_owned(), conflicts as f64),
            ("nthreads".to_owned(), nthreads as f64),
            (
                "conflict_percent".to_owned(),
                100.0 * conflicts as f64 / (nsamples + conflicts) as f64,
            ),
            ("greedy_time".to_owned(), greedy_time.as_secs_f64()),
            ("glauber_time".to_owned(), glauber_time.as_secs_f64()),
        ]
        .iter()
        .cloned()
        .collect(),
    )
}

/// Crucially, only drop neighbor locks after vertex is updated.
/// (whenever the parameter argument is cleared).
fn try_mcmc_update<'a, R: Rng>(
    rng: &mut R,
    colors: &'a [Rwu32],
    graph: &Graph,
    viable_colors: &mut DiscreteSampler,
    neighbor_guards: &mut Vec<ReadGuard<'a>>,
) -> Option<()> {
    viable_colors.reset();
    debug_assert!(neighbor_guards.is_empty());

    let v: u32 = rng.gen_range(0, graph.nvertices() as u32);
    let mut v_color_guard = colors[v as usize].try_write_lock()?;

    for &w in graph.neighbors(v) {
        let (c, neighbor_guard) = colors[w as usize].try_read_lock()?;
        neighbor_guards.push(neighbor_guard);
        viable_colors.remove(c);
        if viable_colors.nalive() == 1 {
            return Some(());
        }
    }

    let chosen = viable_colors.sample(rng);
    v_color_guard.write(chosen as u32);

    Some(())
}

/// A uniform sampler over arbitrary subsets of 0..n which allows:
///
///  - constant-time removal from domain
///  - constant-time sampling
///  - (if it was to be implemented) constant-time insertion
struct DiscreteSampler {
    alive_set: Vec<u32>,
    dead_set: Vec<u32>,
    index: Vec<u32>,
    alive: Vec<bool>,
}

impl DiscreteSampler {
    /// Initializes a uniform sampler over the entire domain 0..n.
    fn new(n: u32) -> Self {
        Self {
            alive_set: (0..n).collect(),
            dead_set: Vec::with_capacity(n as usize),
            index: (0..n).collect(),
            alive: vec![true; n as usize],
        }
    }

    /// No-op if `i` is dead.
    fn remove(&mut self, i: u32) {
        if !self.alive[i as usize] {
            return;
        }
        let ix = self.index[i as usize];
        self.index[*self.alive_set.last().unwrap() as usize] = ix;
        let ii = self.alive_set.swap_remove(ix as usize);
        assert!(ii == i);
        self.index[i as usize] = std::u32::MAX;
        self.dead_set.push(i);
        self.alive[i as usize] = false;
    }

    /// Reverts to original domain.
    fn reset(&mut self) {
        for i in self.dead_set.drain(..) {
            self.index[i as usize] = self.alive_set.len() as u32;
            self.alive_set.push(i);
            self.alive[i as usize] = true;
        }
    }

    /// Samples from alive domain.
    fn sample<R: Rng>(&self, rng: &mut R) -> u32 {
        self.alive_set[rng.gen_range(0, self.alive_set.len())]
    }

    fn nalive(&self) -> usize {
        self.alive_set.len()
    }
}
