//! The core coloring functionality for sparse datasets represented
//! as pages of sparse matrices.

use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::mem;
use std::sync::atomic;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use indicatif::{MultiProgress, ProgressBar, ProgressIterator, ProgressStyle};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::{graph::Graph, graph::Vertex, SparseMatrix};

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

/// Returns `(ncolors, colors)` for a max-degree-ordered coloring of the graph.
pub fn greedy(graph: &Graph) -> (u32, Vec<u32>) {
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

    (adjacent_colors.len() as u32, colors)
}

/// Returns `(ncolors, colors)` for a Glauber-colored graph. If greedy colors
/// requires more colors than specified, then increases the color count to that many.
pub fn glauber(graph: &Graph, ncolors: u32, nsamples: usize) -> (u32, Vec<u32>) {
    unimplemented!()
    // implement parallel monte carlo with *SORTED* neighbors
    // make neighbors atomic u64s
}

// TODO impl greedy() and glauber()
// print collision stats in greedy.rs
// try to improve glauber with paralell mcmcm

/*
    let nfeatures = csr[0].cols();
    let mut greedy = OnlineGreedy::init(nfeatures);

    let sty = ProgressStyle::default_bar()
        .template("{prefix} [{elapsed_precise}] [{bar:40}] {pos:>10}/{len:10} (eta {eta})")
        .progress_chars("##-");
    let outer = ProgressBar::new(nfeatures.try_into().unwrap()).with_style(sty.clone());
    outer.set_prefix("features");
    outer.inc(0);

    // These constants generally keep memory usage under 32GB, they can be modified
    // as necessary for a speed/memory tradeoff. Per thread should be in principle
    // tuned somehow to each dataset so whp there's at least something each thread is
    // aggregating (as opposed to doing no-ops).
    let per_thread = 1024 * 4;
    let max_parallel = 1024 * 4 * per_thread;
    for lo in (0..nfeatures).step_by(max_parallel) {
        let hi = nfeatures.min(lo + max_parallel);
        let mut collisionss = vec![HashMap::<u32, u64>::new(); hi - lo];

        let inner = ProgressBar::new(csr.len().try_into().unwrap()).with_style(sty.clone());
        inner.set_prefix("pages   ");
        inner.inc(0);
        for matrix in csr {
            let nchunks = (hi - lo + per_thread - 1) / per_thread;
            (0..nchunks)
                .into_par_iter()
                .zip(collisionss.par_chunks_mut(per_thread))
                .for_each(|(chunk_ix, collisions)| {
                    let lo_feat = (chunk_ix * per_thread + lo) as u32;
                    let hi_feat = hi.min((chunk_ix + 1) * per_thread + lo) as u32;
                    for row in matrix.outer_iterator() {
                        let col_ixs = row.indices();

                        let ub = match col_ixs.binary_search(&hi_feat) {
                            Ok(x) => x,
                            Err(x) => x,
                        };

                        let lb = match col_ixs[..ub].binary_search(&lo_feat) {
                            Ok(x) => x,
                            Err(x) => x,
                        };

                        for write_ix in lb..ub {
                            for col_ix in &col_ixs[..lb] {
                                *collisions[(col_ixs[write_ix] - lo_feat) as usize]
                                    .entry(*col_ix)
                                    .or_default() += 1;
                            }
                        }
                    }
                });
            inner.inc(1);
        }
        inner.finish_and_clear();

        for collisions in collisionss {
            let nbrs =
                collisions
                    .into_iter()
                    .filter_map(|(nbr, cnt)| if cnt >= k.into() { Some(nbr) } else { None });
            greedy.add(nbrs);
            outer.inc(1);
        }
    }

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
}

impl OnlineGreedy {
    fn init(nfeatures: usize) -> Self {
        Self {
            nfeatures,
            colors: vec![0u32; nfeatures],
            remap: vec![0u32; nfeatures],
            nfeat_per_color: Vec::new(),
            current: 0,
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
    }
}
*/
