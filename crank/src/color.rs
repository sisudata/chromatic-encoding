//! The core coloring functionality for sparse datasets represented
//! as pages of sparse matrices.

use std::convert::TryInto;
use std::iter;
use std::mem;
use std::sync::atomic;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
};

use crate::SparseMatrix;

/// Returns `(colors, remap)` for each feature of the input dataset equivalently represented
/// as a vertical stack of the `csr` and `csc` matrices, after applying the chromatic encoding
/// greedily (using the online coloring for features in the natural ordering defined by the
/// feature index).
///
/// `remap` 1-indexes features with the same color.
pub fn greedy(csr: &[SparseMatrix], csc: &[SparseMatrix], k: u32) -> (Vec<u32>, Vec<u32>) {
    // At a high level, we color a graph where the vertices are features and edges exist between
    // features which co-occurr at least `k` times. This graph is colored by an online serial
    // algorithm which receives adjacency information from every vertex to all previously explored
    // vertices.
    //
    // Adjacency information is communicated via collision counts, where a collision between
    // two features is a row where they co-occurr.
    //
    // Collision counting is parallel for each feature we're measuring collisions with
    // previously explored vertices against, by parallelizing along sparse matrix pages.
    // For each sparse matrix page, we get the rows this feature is present in via CSC
    // matrix and then the colliding features for each of those rows via CSR matrix.
    //
    // We can also parallelize the collision counting, even if the coloring algorithm is serial,
    // we can start counting collisions for the next few vertices while the current one is still
    // completing.
    let nfeatures = csr[0].cols();
    let ref greedy = Mutex::new(OnlineGreedy::init(nfeatures));
    let ref cvar = Condvar::new();

    // TODO: this deadlocks like crazy, need re-entrant mutex
    let mut count = 0;
    let max_parallel = 5;
    iter::from_fn(move || {
        if count == nfeatures {
            return None;
        }

        count += 1;
        let next = count - 1;

        let mut greedy = greedy.lock().unwrap();
        while greedy.current + max_parallel < next {
            greedy = cvar.wait(greedy).unwrap()
        }
        cvar.notify_all();
        Some(next)
    })
    // Parallel bridge is intentional here since the semaphore in the for_each
    // below serializes anyway in order, so it doesn't make sense to start on
    // features which have a high index.
    .par_bridge()
    .map(|f| {
        //print!("starting feat {}\n", f);
        // To keep memory usage low, it's important to cap concurrent collision counters
        // via semaphore (TODO). We're careful not to call rayon under deadlock (rayon/issues/592).

        let collisions: Vec<AtomicU64> = iter::repeat_with(|| 0.into()).take(f).collect();

        csc.par_iter().zip(csr.par_iter()).for_each(|(csc, csr)| {
            csc.outer_view(f)
                .unwrap()
                .iter()
                .map(|(row_ix, _)| row_ix)
                .for_each(|row_ix| {
                    csr.outer_view(row_ix)
                        .unwrap()
                        .iter()
                        .map(|(col_ix, _)| col_ix)
                        .take_while(|&col_ix| col_ix < f)
                        .for_each(|col_ix| {
                            collisions[col_ix].fetch_add(1, Ordering::Relaxed);
                        })
                })
        });

        let nbrs: Vec<usize> = collisions
            .into_par_iter()
            .enumerate()
            .filter_map(|(nbr, collisions)| {
                if collisions.into_inner() >= k.into() {
                    Some(nbr)
                } else {
                    None
                }
            })
            .collect();
        // intentionally separate tasks to allow intermediate parallelism
        (nbrs, f)
    })
    .for_each(|(nbrs, f)| {
        let mut greedy = greedy.lock().unwrap();
        while greedy.current != f {
            greedy = cvar.wait(greedy).unwrap()
        }
        greedy.add(nbrs.into_iter());
        cvar.notify_all();
    });

    let mut greedy = greedy.lock().unwrap();
    greedy.pb.finish_and_clear();
    assert!(greedy.current == greedy.nfeatures);
    (mem::take(&mut greedy.colors), mem::take(&mut greedy.remap))
}

/// Structure to maintain online greedy coloring state.
struct OnlineGreedy {
    nfeatures: usize,
    colors: Vec<u32>,
    /// A vector which maps the original feature index into the rank of that feature
    /// for the color it was assigned (in the ordering of color assignments).
    remap: Vec<u32>,
    nfeat_per_color: Vec<u32>,
    current: usize,
    pb: ProgressBar,
}

impl OnlineGreedy {
    fn init(nfeatures: usize) -> Self {
        Self {
            nfeatures,
            colors: vec![0u32; nfeatures],
            remap: vec![0u32; nfeatures],
            nfeat_per_color: Vec::new(),
            current: 0,
            pb: ProgressBar::new(nfeatures.try_into().unwrap()).with_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{bar:40}] {pos:>10}/{len:10} (eta {eta})")
                    .progress_chars("##-"),
            ),
        }
    }

    fn ncolors(&self) -> usize {
        self.nfeat_per_color.len()
    }

    /// Observe the adjacency for the next vertex, which must be the
    /// `colors.len()`-th. Updates all internal structures to the newly
    /// assigned value for that vertex.
    fn add(&mut self, neighbors: impl Iterator<Item = usize>) {
        assert!(self.current < self.nfeatures);
        let mut is_color_adjacent = vec![false; self.ncolors()];
        let mut nadjacent_colors = 0;

        for nbr in neighbors {
            assert!(
                nbr < self.current,
                "nbr {} >= current {}",
                nbr,
                self.current
            );
            let nbr_color = self.colors[nbr];
            if !is_color_adjacent[nbr_color as usize] {
                is_color_adjacent[nbr_color as usize] = true;
                nadjacent_colors += 1;
                if nadjacent_colors == self.ncolors() {
                    break;
                }
            }
        }

        let chosen = if nadjacent_colors == self.ncolors() {
            self.nfeat_per_color.push(0);
            self.nfeat_per_color.len() - 1
        } else {
            is_color_adjacent.iter_mut().position(|x| !*x).unwrap()
        };

        self.colors[self.current] = chosen as u32;
        self.remap[self.current] = self.nfeat_per_color[chosen] + 1;
        self.nfeat_per_color[chosen] += 1;
        self.current += 1;
        self.pb.inc(1);
    }
}
