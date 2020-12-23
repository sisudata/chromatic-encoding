//! Helper functions for dealing with text data files in
//! a "SIMple SVMlight" (simsvm) format, i.e.,
//! <target> <feature> <feature>...
//! where features should be contiguous non-negative integers.

use std::convert::TryInto;

use rayon::iter::ParallelIterator;

use crate::scanner::{DelimIter, Scanner};
use crate::SparseMatrix;

/// Given a [`DelimIter`] pointing to the front of a line in a
/// simsvm file, this wrapper is a convenient iterator over
/// just the features in that line.
#[derive(Clone)]
pub struct SimSvmLineIter<'a> {
    target: &'a [u8],
    iter: DelimIter<'a>,
}

pub fn stats(scanner: &Scanner) -> DatasetStats {
    let pages: Vec<_> = scanner
        .fold(
            |_| PageStats::default(),
            |mut acc, line| {
                parse(line).for_each(|feature| {
                    acc.nnz += 1;
                    acc.max_feature = acc.max_feature.max(feature);
                });
                acc.nrows += 1;
                acc
            },
        )
        .collect();
    let nfeatures = pages
        .iter()
        .map(|page| page.max_feature)
        .max()
        .map(|x| x + 1)
        .unwrap_or(1)
        .try_into()
        .unwrap();
    DatasetStats { nfeatures, pages }
}

pub struct DatasetStats {
    nfeatures: usize,
    pages: Vec<PageStats>,
}

impl DatasetStats {
    pub fn nfeatures(&self) -> usize {
        self.nfeatures
    }

    pub fn npages(&self) -> usize {
        self.pages.len()
    }

    pub fn nrows(&self) -> usize {
        self.pages.iter().map(|p| p.nrows).sum()
    }

    pub fn avg_nnz(&self) -> f64 {
        self.pages.iter().map(|p| p.nnz).sum::<usize>() as f64 / self.nrows() as f64
    }
}

/// Convert each file to a sparse matrix.
pub fn csr(scanner: &Scanner, stats: &DatasetStats) -> Vec<SparseMatrix> {
    scanner
        .fold(
            |i| {
                (
                    Vec::<u32>::with_capacity(stats.pages[i].nrows + 1),
                    Vec::<u32>::with_capacity(stats.pages[i].nnz),
                )
            },
            |(mut indptr, mut indices), line| {
                let start = indices.len();
                parse(line).for_each(|feature| {
                    assert!(indices.capacity() > indices.len());
                    indices.push(feature)
                });
                indices[start..].sort_unstable();
                assert!(indices[start..].windows(2).all(|s| s[0] < s[1]));
                indptr.push(start.try_into().unwrap());
                (indptr, indices)
            },
        )
        .map(|(mut indptr, indices)| {
            let nnz = indices.len();
            indptr.push(nnz.try_into().unwrap());
            SparseMatrix::new(
                (indptr.len() - 1, stats.nfeatures()),
                indptr,
                indices,
                vec![(); nnz],
            )
        })
        .collect()
}

#[derive(Default)]
struct PageStats {
    max_feature: u32,
    nrows: usize,
    nnz: usize,
}

pub fn parse(mut iter: DelimIter<'_>) -> SimSvmLineIter<'_> {
    let target = iter.next().expect("target");
    SimSvmLineIter { target, iter }
}

impl<'a> Iterator for SimSvmLineIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        self.iter.next().map(|word| {
            let string = std::str::from_utf8(word).expect("utf-8");
            string.parse().expect("parse feature")
        })
    }
}

impl<'a> SimSvmLineIter<'a> {
    pub fn target(&self) -> &'a [u8] {
        self.target
    }
}
