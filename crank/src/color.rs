//! The core coloring functionality for sparse datasets represented
//! as pages of sparse matrices.

use std::convert::TryInto;
use std::sync::atomic;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::SparseMatrix;

/// Returns `(colors, remap)` for each feature of the input dataset equivalently represented
/// as a vertical stack of the `csr` and `csc` matrices, after applying the chromatic encoding
/// greedily (using the online coloring for features in the natural ordering defined by the
/// feature index).
///
/// `remap` 1-indexes features with the same color.
pub fn greedy(csr: &[SparseMatrix], csc: &[SparseMatrix], k: u32) -> (Vec<u32>, Vec<u32>) {
    const COLORS_GUESS: usize = 1024;
    let nfeatures = csr[0].cols();
    let mut colors = vec![0u32; nfeatures];
    let mut remap = vec![0u32; nfeatures];
    let mut nfeat_per_color = Vec::with_capacity(COLORS_GUESS);

    let mut collisions: Vec<AtomicU64> = Vec::with_capacity(nfeatures);
    let mut is_color_adjacent: Vec<AtomicBool> = Vec::with_capacity(COLORS_GUESS);
    for f in 0..nfeatures {
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

        atomic::fence(Ordering::SeqCst); // does rayon do this?
        collisions.par_iter_mut().enumerate().for_each(|(i, x)| {
            if *x.get_mut() >= k.into() {
                is_color_adjacent[colors[i] as usize].store(true, Ordering::Relaxed);
            }
            *x.get_mut() = 0;
        });

        atomic::fence(Ordering::SeqCst); // does rayon do this?
        let maybe_color = is_color_adjacent.iter_mut().position(|x| !*x.get_mut());
        let color = maybe_color.unwrap_or_else(|| {
            nfeat_per_color.push(0);
            is_color_adjacent.push(AtomicBool::new(false));
            is_color_adjacent.len() - 1
        });

        colors[f] = color.try_into().unwrap();
        remap[f] = nfeat_per_color[color as usize] + 1;
        nfeat_per_color[color as usize] += 1;

        collisions.push(AtomicU64::new(0));
        is_color_adjacent
            .iter_mut()
            .for_each(|x| *x.get_mut() = false);
    }
    (colors, remap)
}
