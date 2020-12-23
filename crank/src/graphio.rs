//! Generates a feature co-occurrence graph from a dataset.

use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader, BufWriter};
use std::iter;
use std::mem;
use std::path::PathBuf;
use std::sync::atomic;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use indicatif::{MultiProgress, ProgressBar, ProgressIterator, ProgressStyle};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use serde_json::json;

use crate::{graph::Graph, svmlight, Scanner, SparseMatrix};

/// Returns the number of edges, edge weight avg, and vertex degree avg
/// in a tuple after generating the co-occurrence graph for a design matrix represented
/// as a vertical stack of the `csr` matrices
pub fn write(csr: &[SparseMatrix], nvertices: usize, out: &PathBuf) -> (usize, f64, f64) {
    // The graph here is constructed in an "online" manner, revealing new features as time
    // moves forward.
    //
    // Adjacency information is communicated via collision counts, where a collision between
    // two features is a row where they co-occurr.
    //
    // We can parallelize the collision counting, even if the coloring algorithm is serial,
    // by counting collisions for several few vertices at a time.
    //
    // There's a fancy way of doing this via par_bridge, but that requires re-entrant
    // mutexes to serialize rayon tasks at the consumer side and is generally too messy.
    // Dumber chunk-based parallelism is easier, which is what's implemented below.

    let file = File::create(&out).expect("write file");
    let mut writer = BufWriter::new(file);
    let mut nedges = 0usize;
    let mut edge_weight_sum = 0.0f64;
    let mut vertex_degree_sum = 0.0f64;

    let sty = ProgressStyle::default_bar()
        .template("{prefix} [{elapsed_precise}] [{bar:40}] {pos:>10}/{len:10} (eta {eta})")
        .progress_chars("##-");
    let outer = ProgressBar::new(nvertices.try_into().unwrap()).with_style(sty.clone());
    outer.set_prefix("features");
    outer.inc(0);

    // These constants generally keep memory usage under 32GB, they can be modified
    // as necessary for a speed/memory tradeoff. Per thread should be in principle
    // tuned somehow to each dataset so whp there's at least something each thread is
    // aggregating (as opposed to doing no-ops).
    let per_thread = 1024 * 4;
    let max_parallel = 1024 * 4 * per_thread;
    for lo in (0..nvertices).step_by(max_parallel) {
        let hi = nvertices.min(lo + max_parallel);
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
                            for col_ix in &col_ixs[..write_ix] {
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

        for (offset, collisions) in collisionss.into_iter().enumerate() {
            vertex_degree_sum += collisions.len() as f64;
            let current = lo + offset;
            write!(writer, "{}", current).unwrap();
            for (nbr, cnt) in collisions.into_iter() {
                edge_weight_sum += cnt as f64;
                nedges += 1;
                write!(writer, " {}:{}", nbr, cnt).unwrap();
            }
            writer.write_all(b"\n").unwrap();
            outer.inc(1);
        }
    }

    outer.finish_and_clear();

    (
        nedges,
        edge_weight_sum / nedges as f64,
        vertex_degree_sum / nvertices as f64,
    )
}

/// Reads a single file behind a scanner into an in-memory graph.
pub fn read(scanner: &Scanner, nvertices: usize, min_edge_weight: u32) -> Graph {
    // if you *really* want this to crank then swap out the atomics for sharded owners
    // and use mpsc queues to pass around increment/store messages
    let (offsets, mut edges, offset_time, edge_time) = {
        let mut atomic_offsets: Vec<_> = iter::repeat_with(|| AtomicUsize::new(0))
            .take(nvertices + 1)
            .collect();
        let offset_start = Instant::now();
        scanner
            .fold(
                |_| (),
                |_, line| {
                    let line = svmlight::parse(line);
                    let target = line.target();
                    let neighbors = line
                        .filter(|(_, weight)| *weight >= min_edge_weight)
                        .map(|(neighbor, _)| neighbor);
                    for neighbor in neighbors {
                        atomic_offsets[1 + neighbor as usize].fetch_add(1, Ordering::Relaxed);
                        atomic_offsets[1 + target as usize].fetch_add(1, Ordering::Relaxed);
                    }
                },
            )
            .collect::<()>();
        let offset_time = format!("{:.0?}", Instant::now().duration_since(offset_start));

        let mut cumsum = 0;
        for offset in atomic_offsets.iter_mut() {
            cumsum += *offset.get_mut();
            *offset.get_mut() = cumsum;
        }
        let offsets: Vec<usize> = atomic_offsets
            .iter_mut()
            .map(|offset| *offset.get_mut())
            .collect();

        // technically this is double the number of edges
        let nedges = offsets[offsets.len() - 1];
        let atomic_edges: Vec<_> = iter::repeat_with(|| AtomicU32::new(0))
            .take(nedges)
            .collect();
        let edge_start = Instant::now();
        scanner
            .fold(
                |_| (),
                |_, line| {
                    let line = svmlight::parse(line);
                    let target = line.target();
                    let neighbors = line
                        .filter(|(_, weight)| *weight >= min_edge_weight)
                        .map(|(neighbor, _)| neighbor);
                    for neighbor in neighbors {
                        let target_ix =
                            atomic_offsets[target as usize].fetch_add(1, Ordering::Relaxed);
                        let neighbor_ix =
                            atomic_offsets[neighbor as usize].fetch_add(1, Ordering::Relaxed);
                        atomic_edges[target_ix].store(neighbor, Ordering::Relaxed);
                        atomic_edges[neighbor_ix].store(target, Ordering::Relaxed);
                    }
                },
            )
            .collect::<()>();
        let edge_time = format!("{:.0?}", Instant::now().duration_since(edge_start));
        (
            offsets,
            atomic_edges
                .into_iter()
                .map(|a| a.into_inner())
                .collect::<Vec<_>>(),
            offset_time,
            edge_time,
        )
    };

    let sort_start = Instant::now();
    {
        // fight the borrow checker
        let mut head_and_tail = edges.split_at_mut(0);
        let mut neighbor_lists = Vec::with_capacity(offsets.len() - 1);
        for s in offsets.windows(2) {
            let next_chunk = (s[1] - s[0]) as usize;
            head_and_tail = head_and_tail.1.split_at_mut(next_chunk);
            neighbor_lists.push(head_and_tail.0);
        }
        neighbor_lists
            .par_iter_mut()
            .for_each(|s| s.sort_unstable());
    }
    let sort_time = format!("{:.0?}", Instant::now().duration_since(sort_start));

    println!(
        "{}",
        json!({
            "sort_time": sort_time,
            "edge_time": edge_time,
            "offset_time": offset_time,
        })
    );

    Graph::new(offsets, edges)
}
