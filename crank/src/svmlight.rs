//! Helper functions for dealing with text data files in svmlight format, i.e.,
//! <target> <feature>:<value> <feature>:<value>...
//! where features should be contiguous non-negative integers and same for values
//! (note we don't support float values).

use std::convert::TryInto;
use std::default::Default;
use std::str;

use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;

use crate::scanner::{DelimIter, Scanner};

/// Given a [`DelimIter`] pointing to the front of a line in a
/// simsvm file, this wrapper is a convenient iterator over
/// just the features in that line.
#[derive(Clone)]
pub struct SvmlightLineIter<'a> {
    target: &'a [u8],
    iter: DelimIter<'a>,
}

/// A simple struct containing all the data (but none of the functionality)
/// of a sparse CSR matrix.
#[derive(Default)]
pub struct SparseMatrixComponents {
    pub y: Vec<u32>,
    pub indices: Vec<u32>,
    pub data: Vec<u32>,
    pub indptr: Vec<u64>,
}

/// As above, but with indices/data joined
#[derive(Default)]
struct SparseMatrixComponentsInternal {
    y: Vec<u32>,
    indices_data: Vec<(u32, u32)>,
    indptr: Vec<u64>,
}

/// Convert each file to a sparse matrix's CSR components, returning
/// the target, indices, indptr, and data values.
pub fn csr(scanner: &Scanner) -> SparseMatrixComponents {
    let smc: Vec<_> = scanner
        .fold(
            |_| SparseMatrixComponentsInternal::default(),
            |mut acc, line| {
                let start = acc.indices_data.len();
                let line = parse(line);
                let target = line.target();
                line.for_each(|(feature, value)| {
                    acc.indices_data.push((feature, value));
                });
                acc.indices_data[start..].sort_unstable();
                assert!(acc.indices_data[start..].windows(2).all(|s| s[0] < s[1]));
                acc.indptr.push(start.try_into().unwrap());
                acc.y.push(target);
                acc
            },
        )
        .collect();
    let mut bigbin = smc
        .into_iter()
        .fold(SparseMatrixComponents::default(), |mut acc, x| {
            let offset: u64 = acc.indices.len().try_into().unwrap();
            acc.y.extend(x.y);
            acc.indices.extend(x.indices_data.iter().map(|(x, _)| *x));
            acc.data.extend(x.indices_data.iter().map(|(_, y)| *y));
            acc.indptr.extend(x.indptr.into_iter().map(|i| i + offset));
            acc
        });
    assert!(bigbin.y.len() == bigbin.indptr.len());
    bigbin.indptr.push(bigbin.indices.len().try_into().unwrap());
    assert!(bigbin.indptr.par_windows(2).all(|s| s[0] <= s[1]));
    assert!(bigbin.indptr.par_windows(2).all(|s| {
        let (lo, hi) = (s[0].try_into().unwrap(), s[1].try_into().unwrap());
        bigbin.indices[lo..hi].windows(2).all(|ss| ss[0] < ss[1])
    }));
    bigbin
}

pub fn parse(mut iter: DelimIter<'_>) -> SvmlightLineIter<'_> {
    let target = iter.next().expect("target");
    SvmlightLineIter { target, iter }
}

impl<'a> Iterator for SvmlightLineIter<'a> {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|word| {
            let string = str::from_utf8(word).expect("utf-8");
            let (feature, value) = string
                .rfind(':')
                .map(|pos| (&string[..pos], &string[pos + 1..]))
                .expect("feature-value pair");
            (
                feature.parse().expect("parse feature"),
                value.parse().expect("parse value"),
            )
        })
    }
}

impl<'a> SvmlightLineIter<'a> {
    pub fn target(&self) -> u32 {
        let string = str::from_utf8(self.target).expect("utf-8");
        string.parse().expect("target parse")
    }
}
