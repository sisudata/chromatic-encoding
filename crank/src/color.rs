//! The core coloring functionality for sparse datasets represented
//! as pages of sparse matrices.

use std::collections::HashMap;
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
use rayon::slice::ParallelSlice;

use crate::SparseMatrix;

/// Returns `(colors, remap)` for each feature of the input dataset
/// as a vertical stack of the `csr` matrices, after applying the chromatic encoding
/// greedily (using the online coloring for features in the natural ordering defined by the
/// feature index).
///
/// `remap` 1-indexes features with the same color.
pub fn greedy(csr: &[SparseMatrix], k: u32) -> (Vec<u32>, Vec<u32>) {
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
    // completing. There's a fancy way of doing this via par_bridge, but that requires re-entrant
    // mutexes to serialize rayon tasks at the consumer side and is generally too messy.
    // Dumber chunk-based parallelism is easier.
    let nfeatures = csr[0].cols();
    let mut greedy = OnlineGreedy::init(nfeatures);

    let max_parallel = 32; // rayon num threads?
    for lo in (0..nfeatures).step_by(max_parallel) {
        let hi = nfeatures.min(lo + max_parallel);
        let mut collisionss = vec![HashMap::<u32, u64>::new(); hi - lo];
        for matrix in csr {
            // TODO: parallel this across csr mats?
            let upper_bounds: Vec<_> = matrix
                .indptr()
                .par_windows(2)
                .map(|lohi| {
                    let col_ixs = &matrix.indices()[(lohi[0] as usize)..(lohi[1] as usize)];
                    match col_ixs.binary_search(&(hi as u32)) {
                        Ok(x) => x + 1,
                        Err(x) => x,
                    }
                })
                .collect();
            (lo..hi)
                .into_par_iter()
                .zip(collisionss.par_iter_mut())
                .for_each(|(f, collisions)| {
                    let f = f as u32;
                    for (row, ub) in matrix.outer_iterator().zip(upper_bounds.iter().copied()) {
                        let col_ixs = row.indices();
                        let mut new_ub = 0;
                        for i in (0..ub).rev() {
                            if col_ixs[i] == f {
                                new_ub = i;
                                break;
                            }
                            if col_ixs[i] < f {
                                break;
                            }
                        }
                        for col_ix in &col_ixs[..new_ub] {
                            *collisions.entry(*col_ix).or_default() += 1;
                        }
                    }
                })
        }

        for collisions in collisionss {
            let nbrs =
                collisions
                    .into_iter()
                    .filter_map(|(nbr, cnt)| if cnt >= k.into() { Some(nbr) } else { None });
            greedy.add(nbrs)
        }
    }

    greedy.pb.finish_and_clear();
    assert!(greedy.current == greedy.nfeatures);
    (greedy.colors, greedy.remap)
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
    fn add(&mut self, neighbors: impl Iterator<Item = u32>) {
        // par impl for this in kdd12
        assert!(self.current < self.nfeatures);
        let mut is_color_adjacent = vec![false; self.ncolors()];
        let mut nadjacent_colors = 0;

        for nbr in neighbors {
            assert!(
                (nbr as usize) < self.current,
                "nbr {} >= current {}",
                nbr,
                self.current
            );
            let nbr_color = self.colors[nbr as usize];
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
