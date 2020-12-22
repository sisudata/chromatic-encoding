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

use indicatif::{MultiProgress, ProgressBar, ProgressIterator, ProgressStyle};
use rand_pcg::Lcg64Xsh32;
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
    let (greedy_ncolors, colors, _) = greedy(graph);
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
        .map(|x| AtomicU32::new(x))
        .collect::<Vec<_>>();
    let nthreads = 4; // rayon::current_num_threads() as usize;

    let conflicts = (0..nthreads)
        .into_par_iter()
        .map(|i| {
            let nsamples = (nsamples / nthreads).max(1);
            let mut rng = Lcg64Xsh32::new(0xcafef00dd15ea5e5, i as u64);
            let mut adjacent_color: Vec<bool> = vec![false; ncolors as usize];
            let mut neighbor_colors: Vec<u32> = Vec::new();
            let mut conflicts = 0;

            for _ in 0..nsamples {
                // loop invariant is that none of adjacent_color elements are true

                let reservation = (ncolors + 1) as u32;

                loop {
                    let v: u32 = rng.gen_range(0, graph.nvertices() as u32);

                    // "claim" this vertex
                    let prev = colors[v as usize].swap(reservation, Ordering::Relaxed);
                    if prev == reservation {
                        conflicts += 1;
                        continue;
                    }

                    let mut nadjacent_colors = 0;
                    let mut fail = v;
                    for &w in graph.neighbors(v) {
                        // lock
                        let c = colors[w as usize].swap(reservation, Ordering::Relaxed);
                        if c == reservation {
                            fail = w;
                            break;
                        }
                        if !adjacent_color[c as usize] {
                            nadjacent_colors += 1;
                            adjacent_color[c as usize] = true;
                        }
                        neighbor_colors.push(c);
                    }
                    if fail != v {
                        // unlock
                        colors[v as usize].store(prev, Ordering::Relaxed);
                        graph
                            .neighbors(v)
                            .iter()
                            .take_while(|&&w| w != fail)
                            .zip(neighbor_colors.iter())
                            .for_each(|(&w, &c)| colors[w as usize].store(c, Ordering::Relaxed));
                        conflicts += 1;
                        for i in adjacent_color.iter_mut() {
                            *i = false;
                        }
                        neighbor_colors.clear();
                        continue;
                    }

                    // attempt to get lucky once
                    let mut chosen = rng.gen_range(0, ncolors);
                    if adjacent_color[chosen as usize] {
                        chosen = adjacent_color
                            .iter()
                            .copied()
                            .enumerate()
                            .filter(|(_, x)| !x)
                            .map(|(i, _)| i as u32)
                            .nth(rng.gen_range(0, (ncolors as usize) - nadjacent_colors))
                            .expect("nth");
                    }

                    graph
                        .neighbors(v)
                        .iter()
                        .zip(neighbor_colors.iter())
                        .for_each(|(&w, &c)| colors[w as usize].store(c, Ordering::Relaxed));

                    colors[v as usize].store(chosen as u32, Ordering::Relaxed);

                    for i in adjacent_color.iter_mut() {
                        *i = false;
                    }
                    neighbor_colors.clear();
                    break;
                }
            }
            conflicts
        })
        .sum::<usize>();

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
                "conflict_rate".to_owned(),
                100.0 * conflicts as f64 / (nsamples + conflicts) as f64,
            ),
        ]
        .iter()
        .cloned()
        .collect(),
    )
}
