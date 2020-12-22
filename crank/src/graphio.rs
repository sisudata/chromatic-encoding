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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use indicatif::{MultiProgress, ProgressBar, ProgressIterator, ProgressStyle};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

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
    let mut offsets = scanner.fold_serial(vec![0usize; nvertices + 1], |mut degree, line| {
        let line = svmlight::parse(line);
        let target = line.target();
        line.filter(|(_, weight)| *weight >= min_edge_weight)
            .for_each(|(neighbor, _)| {
                degree[1 + neighbor as usize] += 1;
                degree[1 + target as usize] += 1;
            });
        degree
    });
    let mut cumsum = 0;
    for offset in offsets.iter_mut() {
        cumsum += *offset;
        *offset = cumsum;
    }
    // technically double the number of edges
    let nedges = offsets[offsets.len() - 1];
    let mut positions = offsets.clone();
    let edges = scanner.fold_serial(vec![0u32; nedges], move |mut edges, line| {
        let line = svmlight::parse(line);
        let target = line.target();
        let neighbors = line
            .filter(|(_, weight)| *weight >= min_edge_weight)
            .map(|(neighbor, _)| neighbor);
        for neighbor in neighbors {
            edges[positions[target as usize]] = neighbor;
            edges[positions[neighbor as usize]] = target;
            positions[target as usize] += 1;
            positions[neighbor as usize] += 1;
        }
        edges
    });
    Graph::new(offsets, edges)
}
