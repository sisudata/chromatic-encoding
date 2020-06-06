//! The `feature` module controls featurization from the raw u8
//! input.
//!
//! In a multi-pass setting, it doesn't really make sense to use
//! hashing trick at all. Hashing trick has two purposes:
//!
//! 1. Dimensionality reduction while preserving linear inner products,
//! which is great for simulating kernel methods by using featurized
//! sparse vectors.
//!
//! 2. Reducing RAM usage by not storing feature strings.
//!
//! (1) is addressed differently by chromatic sparse learning, through
//! mutual exclusivity.
//!
//! (2) can be fixed without such large collision costs by using a swiss
//! table that stores the larger 64-bit hash of the input string. Collisions
//! may still happen, but much more rarely.

// vw uses murmur3, but https://github.com/rurban/smhasher/ shows
// other hashes may do better.
use crate::identity::{Identity64, IdentityHasher};
use crate::svm_scanner::DelimIter;
use bstr::ByteSlice;
use fasthash::xx;
use std::collections::HashMap;
use std::convert::TryInto;
use std::hash::BuildHasherDefault;

type IdentityHashMap<T> = HashMap<Identity64, T, BuildHasherDefault<IdentityHasher>>;

pub(crate) struct FeaturizerConstructor {
    // data for svm inputs (hashes of features)
    counts: IdentityHashMap<usize>,
    lines: usize,

    // data for tsv inputs (hash for categorical, both optional at the
    // same time, yes i know this is a little sloppy and should be an enum).
    is_dense: Option<Vec<bool>>,
    // for columns which are !is_dense, this contains cross-column
    // feature counts.
    categorical_counts: Vec<IdentityHashMap<usize>>,

    // minimum frequency for a feature to appear
    freq_cutoff: usize,
}

impl FeaturizerConstructor {
    /// Factorization behavior varies between tabular tsv and sparse
    /// svm input. See the flag of the name `tsv_dense` for details in
    /// main.rs.
    ///
    /// Features appearing less than `freq_cutoff` times will be discarded.
    pub(crate) fn new(freq_cutoff: usize, tsv_dense: Option<Vec<usize>>) -> Self {
        let is_dense = tsv_dense.map(|dense_counts| {
            let largest = dense_counts.iter().max().copied().unwrap_or(0) + 1;
            let mut bitmap = vec![false; largest];
            for i in dense_counts.into_iter() {
                bitmap[i] = true;
            }
            bitmap
        });

        FeaturizerConstructor {
            counts: Default::default(),
            lines: 0,
            is_dense,
            categorical_counts: Vec::new(),
            freq_cutoff,
        }
    }

    fn make_sure_room(&mut self, i: usize) {
        while self.categorical_counts.len() <= i {
            self.categorical_counts.push(Default::default());
        }
    }

    /// Given a delimiter pointing to the start of an SVMLight line,
    /// counts features in the line. Returns the total number of words
    /// in that line.
    pub(crate) fn read_words(&mut self, line: DelimIter<'_>) -> usize {
        self.lines += 1;
        let line = line.skip(1); // skip the target
        if self.is_dense.is_some() {
            // tsv input
            let mut count = 0;
            for (i, word) in line.enumerate() {
                let is_dense = self.is_dense.as_ref().unwrap();
                if !is_dense.get(i).copied().unwrap_or(false) {
                    self.make_sure_room(i);
                    *self.categorical_counts[i]
                        .entry(Identity64::from(xx::hash64(word)))
                        .or_insert(0) += 1;
                }
                count += 1;
            }
            count
        } else {
            // svm input
            line.filter(|word| not_space(*word))
                .map(strip_value)
                .map(|word| {
                    *self
                        .counts
                        .entry(Identity64::from(xx::hash64(word)))
                        .or_insert(0) += 1;
                })
                .count()
        }
    }

    fn leftmerge(left: &mut IdentityHashMap<usize>, right: &mut IdentityHashMap<usize>) {
        if left.len() < right.len() {
            std::mem::swap(left, right)
        }
        for (&k, &v) in right.iter() {
            *left.entry(k).or_insert(0) += v;
        }
    }

    pub(crate) fn merge(&mut self, other: &mut Self) {
        Self::leftmerge(&mut self.counts, &mut other.counts);
        self.lines += other.lines;
        assert!(self.is_dense == other.is_dense);
        if self.categorical_counts.len() < other.categorical_counts.len() {
            std::mem::swap(&mut self.categorical_counts, &mut other.categorical_counts);
        }
        for (left, right) in self
            .categorical_counts
            .iter_mut()
            .zip(other.categorical_counts.iter_mut())
        {
            Self::leftmerge(left, right);
        }
    }

    pub(crate) fn build(self) -> Featurizer {
        let freq_cutoff = self.freq_cutoff;
        if let Some(is_dense) = self.is_dense {
            let mut sparse_features: Vec<_> = self
                .categorical_counts
                .into_iter()
                .enumerate()
                .flat_map(|(i, count_map)| {
                    count_map
                        .into_iter()
                        .filter(move |(_, count)| *count >= freq_cutoff)
                        .map(move |(hash, count)| (count, i, hash))
                })
                .collect();
            sparse_features.sort_unstable();
            sparse_features.reverse();

            let sz = sparse_features
                .iter()
                .map(|(_, i, _)| i)
                .max()
                .copied()
                .unwrap_or(0)
                + 1;
            let mut feature_idxs = vec![IdentityHashMap::<u32>::default(); sz];

            let mut ctr = 0u32;
            for (_, i, hash) in sparse_features.iter() {
                feature_idxs[*i].insert(*hash, ctr);
                ctr += 1;
            }

            let sparse_counts: Vec<_> = sparse_features
                .into_iter()
                .map(|(count, _, _)| count)
                .collect();

            assert!(feature_idxs.iter().map(|m| m.len()).sum::<usize>() == sparse_counts.len());
            assert!(feature_idxs.len() >= is_dense.iter().filter(|&x| *x).count());

            return Featurizer {
                sparse: Default::default(),
                dense: Default::default(),
                is_dense: Some(is_dense),
                feature_idxs,
                sparse_counts,
            };
        }

        // POTENTIAL ALGORITHMIC IMPROVEMENT
        //
        // Make the threshold for what's considered sparse a flag.
        // Also record any features which have a non-1.0 value.
        // If some decent proportion of lines contain the feature with
        // a non-1.0 value, consider it dense too.
        let lines = self.lines / 10;
        let (mut dense_features, mut sparse_features) = self
            .counts
            .into_iter()
            .filter(move |(_, count)| *count >= freq_cutoff)
            .partition::<Vec<_>, _>(|(_, count)| *count >= lines);
        dense_features.sort_unstable_by_key(|(_, count)| *count);
        dense_features.reverse();
        sparse_features.sort_unstable_by_key(|(_, count)| *count);
        sparse_features.reverse();
        let nsparse = sparse_features.len();
        let sparse_counts: Vec<_> = sparse_features.iter().map(|(_, count)| *count).collect();
        Featurizer {
            sparse: sparse_features
                .into_iter()
                .enumerate()
                .map(|(i, (hash, _))| (hash, i as u32))
                .collect(),
            dense: dense_features
                .into_iter()
                .enumerate()
                .map(|(i, (hash, _))| (hash, (i + nsparse) as u32))
                .collect(),
            is_dense: None,
            feature_idxs: Vec::default(),
            sparse_counts,
        }
    }
}

pub(crate) struct Featurizer {
    // params for svm input

    // sparse features are missing in at least one row. note their values are ignored.
    sparse: IdentityHashMap<u32>,
    // dense features must occur in every row
    dense: IdentityHashMap<u32>,

    // params for tsv input
    is_dense: Option<Vec<bool>>,
    // for columns which are !is_dense, this contains categorical column value
    // to global cross-column feature index mapping.
    feature_idxs: Vec<IdentityHashMap<u32>>,

    // present in both
    // vector from sparse feature index -> its counts
    sparse_counts: Vec<usize>,
}

impl Featurizer {
    pub(crate) fn nsparse(&self) -> usize {
        self.sparse_counts.len()
    }

    pub(crate) fn ndense(&self) -> usize {
        if let Some(is_dense) = &self.is_dense {
            is_dense.iter().copied().filter(|x| *x).count()
        } else {
            self.dense.len()
        }
    }
    pub(crate) fn istsv(&self) -> bool {
        self.is_dense.is_some()
    }

    /// For tsv only, computes the "free" coloring we get
    /// from each sparse input column.
    pub(crate) fn tsvcolors(&self) -> (Vec<u32>, u32) {
        assert!(self.istsv());
        let is_dense = self.is_dense.as_ref().unwrap();
        let mut colors = vec![std::u32::MAX; self.nsparse()];
        let ncolors: u32 = (self.feature_idxs.len() - self.ndense())
            .try_into()
            .unwrap();
        let mut ctr = 0;
        for (i, features) in self.feature_idxs.iter().enumerate() {
            let dense = is_dense.get(i).copied().unwrap_or(false);
            if dense {
                continue;
            }
            for &feature_idx in features.values() {
                let feature_idx = feature_idx as usize;
                assert!(colors[feature_idx] == std::u32::MAX);
                colors[feature_idx] = ctr;
            }
            ctr += 1;
        }
        assert!(ctr == ncolors);
        assert!(colors.iter().max().unwrap() + 1 == ncolors as u32);
        (colors, ncolors)
    }

    /// Only applicable to sparse colors (index below nsparse())
    /// Note features indices are sorted in descending order by count.
    pub(crate) fn feature_count(&self, feature_idx: u32) -> usize {
        self.sparse_counts[feature_idx as usize]
    }

    /// If the given word is a sparse feature, returns its feature index, else None.
    /// The index should be the index of the feature in the line (used for tsv).
    pub(crate) fn sparse(&self, i: usize, word: &[u8]) -> Option<u32> {
        if let Some(is_dense) = &self.is_dense {
            if is_dense.get(i).copied().unwrap_or(false) {
                return None;
            }
            let hash = Identity64::from(xx::hash64(word));
            return self.feature_idxs[i].get(&hash).copied();
        }
        if !not_space(word) {
            return None;
        }
        let word = strip_value(word);
        let hash = Identity64::from(xx::hash64(word));
        self.sparse.get(&hash).copied()
    }

    /// Given an iterator over an SVMLight line (or even a subsequence
    /// of it), write out dense features in it into an output array of
    /// size ndense, writes the i+nsparse()-th dense feature
    /// observed in the iterator into the i-th position in the array.
    ///
    /// Any dense features not present in the iteration are set to 0.
    ///
    /// For tsv input, this assumes we're reading the tsv starting at the
    /// first feature column and records dense outputs accordingly.
    pub(crate) fn write_dense<'a>(&self, line: impl Iterator<Item = &'a [u8]>, out: &mut [f64]) {
        for f in out.iter_mut() {
            *f = 0.0;
        }
        if let Some(is_dense) = &self.is_dense {
            let mut ctr = 0;
            line.zip(is_dense.iter().copied())
                .for_each(|(word, dense)| {
                    if dense {
                        let value: f64 = if word.len() > 0 {
                            let value_str = std::str::from_utf8(word).expect("utf-8 value (tsv)");
                            str::parse(value_str).expect("f64 parse (tsv)")
                        } else {
                            0.
                        };
                        out[ctr] = value;
                        ctr += 1;
                    }
                });
            return;
        }

        line.filter(|word| not_space(*word))
            .map(pair_value)
            .flat_map(|(feature, value)| {
                let hash = Identity64::from(xx::hash64(feature));
                let idx = self.dense.get(&hash)?;
                let value_str = std::str::from_utf8(value).expect("utf-8 value (svm)");
                let value: f64 = str::parse(value_str).expect("f64 parse (svm)");
                Some(((*idx as usize) - self.nsparse(), value))
            })
            .for_each(|(idx, value)| out[idx] = value);
    }
}

fn strip_value(word: &[u8]) -> &[u8] {
    word.rfind_byte(b':')
        .map(|pos| &word[..pos])
        .unwrap_or(word)
}

pub fn pair_value(word: &[u8]) -> (&[u8], &[u8]) {
    word.rfind_byte(b':')
        .map(|pos| (&word[..pos], &word[pos + 1..]))
        .unwrap_or_else(|| (word, b"1"))
}

pub fn not_space(word: &[u8]) -> bool {
    !word.iter().copied().all(|c| c.is_ascii_whitespace())
}
