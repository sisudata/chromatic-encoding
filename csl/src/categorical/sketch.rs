//! Shared routine for mergeable, updateable feature statistics sketches.

// POTENTIAL OPTIMIZATION
//
// There's some shitty anti-scalability here not present in other parts
// of the code on a high-memory machine.
//
// Here's Malicious URLs data for feature stats time
//  8 threads  7 sec
// 16 threads  7 sec
// 32 threads 15 sec
// 64 threads 26 sec
//
// It's strange that the above behavior doesn't match up with the time for
// the first cardinality scan above, which has a pattern of
// 592ms, 300ms, 151ms, 112ms for 8, 16, 32, 64 threads, respectively
//
// So it doesn't feel like the culprit is overly-parallelized SSD reads
// from the mmap since they do fine the first pass.
// Compounding this hypothesis is that
// the other passes (graph construction, writing results) scale well.
//
// My guess is that reduce is shittily span-bounded parallel computation, where
// the reduce tree gets bottlnecked with feature scans at the top.
//
// This can be fixed by transposing the fold/reduce step:
//
// Folder 1 generates [stats array len N]
// Folder 2 generates [stats array len N] for another disjoint region
// Folder 3 ...
// Folder 4 ...
//
// Instead of rayon's shitty diadic reduces, which look like
// reduce(reduce(fold1, fold2), reduce(fold3, fold4)),
//
// just collect the folds into a matrix, maybe even having the folders
// write into a transpose of the following matrix directly (not pictured)
//
// [ <-- fold1 --> ]
// [ <-- fold2 --> ]
// [ <-- fold3 --> ]
// [ <-- fold4 --> ]
//
// Then do a vectorized reduce over the columns of this matrix (cf. rows of its
// transpose).

use crate::feature::Featurizer;
use crate::svm_scanner::{DelimIter, SvmScanner};
use std::fmt::Display;

pub(crate) trait Sketch: Default + Clone + Send + Sync {
    fn update(&mut self, target: f64);

    fn merge(&mut self, other: &Self);

    // Consider a merge_all(sketches: &[Self]) -> Self
    // See end of file for discussion.
}

#[derive(Default, Clone)]
struct LineSketch<S> {
    most_recent_line: usize,
    sketch: S,
}

impl<S: Sketch> LineSketch<S> {
    fn update(&mut self, line_num: usize, target: f64) {
        if self.most_recent_line == line_num {
            return;
        }
        self.most_recent_line = line_num;
        self.sketch.update(target)
    }

    fn merge(&mut self, other: &Self) {
        self.sketch.merge(&other.sketch);
    }
}

/// Map reduce to collect per-feature statistics over lines.
pub(crate) fn map_reduce_sketch<S, F>(
    train: &SvmScanner,
    featurizer: &Featurizer,
    filter: F,
) -> (S, Vec<S>)
where
    S: Sketch,
    F: for<'a> Fn(DelimIter<'a>) -> bool + Sync + Send,
{
    // add in an extra "global" feature at the end for overall numbers.
    let global = featurizer.nsparse();
    let (_, stats) = train.fold_reduce(
        || (0usize, vec![LineSketch::<S>::default(); global + 1]),
        |(mut line_num, mut stats), mut word_iter| {
            if !filter(word_iter.clone()) {
                return (line_num, stats);
            }
            line_num += 1;
            // SAFETY: caller promised the first word is f64 in ascii
            // in SvmScanner guarantees
            let target: &str =
                unsafe { std::str::from_utf8_unchecked(word_iter.next().expect("target")) };
            let target: f64 = target.parse().expect("float format");
            for h in word_iter
                .enumerate()
                .flat_map(|(i, word)| featurizer.sparse(i, word))
            {
                let stat = &mut stats[h as usize];
                stat.update(line_num, target);
            }
            let stat = &mut stats[global];
            stat.update(line_num, target);
            (line_num, stats)
        },
        |(_, mut l_stats), (_, r_stats)| {
            for (l_stat, r_stat) in l_stats.iter_mut().zip(r_stats.iter()) {
                l_stat.merge(r_stat);
            }
            (0, l_stats)
        },
    );

    let mut all: Vec<_> = stats
        .into_iter()
        .map(|line_sketch| line_sketch.sketch)
        .collect();
    let global = all.pop().unwrap();
    (global, all)
}

/// Misc utility function for printing quantiles.
pub(crate) fn pretty_stats<T: PartialOrd + Display>(mut v: Vec<T>) -> String {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    [
        0000, 25000, 50000, 75000, 90000, 95000, 99000, 99900, 99990, 99999, 100000,
    ]
    .iter()
    .copied()
    .map(|p| format!("{}%: {}", p as f64 / 1000., v[(p * (v.len() - 1) / 100000)]))
    .collect::<Vec<String>>()
    .join(" ")
}
