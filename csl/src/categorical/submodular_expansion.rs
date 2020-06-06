//! Submodular quantization encoder.
//! Based on https://arxiv.org/abs/1904.13389
//!
//! A few practical changes are made:
//!
//! 1. enabling inputs with conditional label probabilities of 0 or 1
//! (original algorithm only allows for conditional probability inside
//! of the open interval (0, 1)).
//! 2. a forced "NULL" value is always uniquely represented to recover
//! absences with full fidelity, even after quantization.
//! 3. lazy greedy instead of stochastic lazy greedy is used.
//! 4. original paper underspecifies the global budgeting strategy in
//! Section 3.1, which is a shame since it's the only part that doesn't
//! have guarantees and thus can't be backed out. I had to guess what something
//! reasonable would be; it's a weak point in the algorithm's definition.

use crate::categorical::sketch;
use crate::feature::Featurizer;
use crate::svm_scanner::DelimIter;
use crate::svm_scanner::SvmScanner;
use fasthash::XXHasher;
use ordered_float::NotNan;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::convert::TryInto;
use std::default::Default;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::time::Instant;

// POTENTIAL ALGORITHMIC IMPROVEMENT
//
// Rather than performing submodular optimization for identifying feature
// vocabularies with maximal mutual information with the output bit
// "target <= 0", we can use Munro-Paterson to figure out the median
// of target and use that for feature prioritization.
//
// POTENTIAL ALGORITHMIC IMPROVEMENT
//
// Submodular, globally done. Across feature boundaries, just
// enforce cutoffs.

#[derive(Default, Clone, Debug)]
struct Sketch {
    // How many times does the feature appear with a target > 0?
    positives: usize,
    // How many times does the feature appear with a target <= 0?
    negatives: usize,
}

pub(crate) struct EncodingDictionary {
    // number of generated features. see `num_quanta` in Encoder
    num_quanta: usize,
    // maps original feature index to new one, many-to-one-or-nothing.
    quantization: Vec<Option<u32>>,
    // maps the null indicator to a new feature index (or nothing)
    // null indicators are set when no features associated with a color
    // are present in the example
    null_quanta: Vec<Option<u32>>,
    // colors for each original feature index
    colors: Vec<u32>,
    ncolors: usize,
    split_rate: usize,
}

pub(crate) struct Encoder {
    // Number of features that will get generated. Except in rare cases, this
    // will end up being the budget.
    num_quanta: usize,
    // all features are 1-valued
    active: Vec<u32>, // list of indices of active quantized features
    // i-th position in color_index corresponds to the location in
    // `active` where the i-th color covariate's active feature
    // resides.
    color_index: Vec<Option<u32>>,
    // a copy of the encoding dictionary's null quanta
    null_quanta: Vec<Option<u32>>,
    // see main.rs for definition
    split_rate: usize,
}

impl sketch::Sketch for Sketch {
    fn update(&mut self, target: f64) {
        let target = if target > 0.0 { 1 } else { 0 };
        self.positives += target;
        self.negatives += 1 - target;
    }

    fn merge(&mut self, other: &Self) {
        self.positives += other.positives;
        self.negatives += other.negatives;
    }
}

impl EncodingDictionary {
    pub(crate) fn new(
        budget: usize,
        train: &SvmScanner,
        featurizer: &Featurizer,
        ncolors: usize,
        colors: Vec<u32>,
        split: bool, // whether to do hash-based splitting for data
        sort: bool,  // whether to do a global sort instead of globally maximizing mutual info sum
        split_rate: usize,
    ) -> Self {
        let split_rate = split_rate as usize;
        let start = Instant::now();
        let (global, features) =
            sketch::map_reduce_sketch::<Sketch, _>(train, featurizer, |line| {
                !split || skip_example(line, split_rate)
            });
        println!(
            "sketch collection {:.0?}",
            Instant::now().duration_since(start)
        );
        println!(
            "counts for feature x: {}",
            sketch::pretty_stats(
                features
                    .iter()
                    .map(|sketch| sketch.positives + sketch.negatives)
                    .collect()
            )
        );

        println!(
            "P(y>0|x), in %: {}",
            sketch::pretty_stats(
                features
                    .iter()
                    .map(|sketch| if sketch.positives + sketch.negatives > 0 {
                        (sketch.positives * 100 / (sketch.positives + sketch.negatives)) as f64
                            / 100.0
                    } else {
                        0.0
                    })
                    .collect()
            )
        );

        // di == dictionaries, defined by btrees
        let (mi, di) = MutualInformation::new(global, features, &colors, ncolors);

        let start = Instant::now();
        let (dictionary, null_dict, num_quanta) = quantize(mi, di, budget, sort);
        println!(
            "submodular quantization {:.0?}",
            Instant::now().duration_since(start)
        );

        EncodingDictionary {
            num_quanta,
            colors: colors,
            quantization: dictionary,
            null_quanta: null_dict,
            ncolors,
            split_rate,
        }
    }
}

impl Encoder {
    pub(crate) fn new(dictionary: &EncodingDictionary) -> Self {
        Encoder {
            num_quanta: dictionary.num_quanta,
            active: Default::default(),
            color_index: vec![None; dictionary.ncolors],
            null_quanta: dictionary.null_quanta.clone(),
            split_rate: dictionary.split_rate,
        }
    }

    pub(crate) fn skip_example(&self, d: DelimIter<'_>) -> bool {
        skip_example(d, self.split_rate)
    }

    pub(crate) fn observe(&mut self, dictionary: &EncodingDictionary, feature: u32) {
        let h = feature as usize;
        let c = dictionary.colors[h] as usize;

        let active = match dictionary.quantization[h] {
            None => return,
            Some(ix) => ix,
        };

        let curr_idx = match self.color_index[c] {
            None => {
                self.color_index[c] = Some(self.active.len().try_into().unwrap());
                self.active.push(active);
                return;
            }
            Some(x) => x as usize,
        };

        // Features can overlap in the validation set, but we still want to be
        // deterministic in our outputs, regardless of feature order.
        //
        // Favor higher quantized values.
        if active <= self.active[curr_idx] {
            return;
        }

        self.active[curr_idx] = active;
    }

    pub(crate) fn dense_offset(&self) -> usize {
        self.num_quanta
    }

    pub(crate) fn finish<W: Write>(&mut self, writer: &mut W) {
        for i in &self.active {
            write!(writer, " {}:1", i).expect("successful write");
        }
        self.active.clear();
        for i in 0..self.color_index.len() {
            match (self.null_quanta[i], self.color_index[i]) {
                (Some(null_feature), None) => {
                    write!(writer, " {}:1", null_feature).expect("successful write");
                }
                _ => {}
            }
            self.color_index[i] = None
        }
    }
}

/// Structure maintains the metadata for defining a quantization of
/// multiple input categorical variables and a target binary variable.
///
/// The input categorical variable are sequentially arranged
/// in the `cfc` array, with `ncolors` disjoint contiguous chunks
/// defined by sequential segments `co[i]:co[i+1]` in the `ncolors+1`-sized
/// `co` color offsets array.
///
/// Within each segment, the individual categorical variables label
/// statistics have been sorted by values v, ordered by their
/// P(Y=1|X=v) conditional positive probability. Then, these segments
/// were accumulated by sum.
struct MutualInformation {
    ncolors: usize,
    nfeatures: usize,
    // cumulative feature counts
    cfc: Vec<CumulativeFeature>,
    // color offsets
    co: Vec<usize>,
    global: Sketch,
}

/// Represents the information associated with an attained
/// feature x in the training set. That feature was assigned a fixed
/// color.
#[derive(Debug)]
struct InputFeature {
    positives: usize,
    negatives: usize,
    color: u32,
    feature_idx: u32,
}

impl InputFeature {
    fn count(&self) -> usize {
        self.positives + self.negatives
    }
}

/// Contains cumulative counts for fast submodular function processing
/// of the above. Used in an array.
#[derive(Default, Clone)]
struct CumulativeFeature {
    cumulative_negatives: usize,
    cumulative_count: usize,
    feature_idx: u32,
}

impl MutualInformation {
    fn new(
        global: Sketch,
        features: Vec<Sketch>,
        colors: &[u32],
        ncolors: usize,
    ) -> (Self, Vec<BTreeSet<usize>>) {
        assert!(features.len() == colors.len());
        assert!(colors.iter().max().unwrap() + 1 == ncolors as u32);

        if cfg!(debug_assertions) {
            let mut color_sketches = vec![Sketch::default(); ncolors];
            for (f, &c) in features.iter().zip(colors.iter()) {
                color_sketches[c as usize].positives += f.positives;
                color_sketches[c as usize].negatives += f.negatives;
            }

            for (i, color_sketch) in color_sketches.iter().enumerate() {
                debug_assert!(
                    color_sketch.positives <= global.positives,
                    "positives over-counted for color {} stats {:?} global {:?}",
                    i,
                    color_sketch,
                    global
                );
                debug_assert!(
                    color_sketch.negatives <= global.negatives,
                    "negatives over-counted for color {} stats {:?} global {:?}",
                    i,
                    color_sketch,
                    global
                );
            }
        }

        let start = Instant::now();

        // collect per-feature counts with color information beside it
        let mut features: Vec<_> = features
            .into_iter()
            .zip(colors)
            .enumerate()
            .map(|(i, (sketch, &color))| InputFeature {
                positives: sketch.positives,
                negatives: sketch.negatives,
                feature_idx: i.try_into().unwrap(),
                color,
            })
            .collect();

        let num_nonnull_features: usize = features.len();

        // add "null" features at feature index nfeatures + i which are sentinel values
        // for situations where the i-th color has no active feature.
        features.extend((0..ncolors).map(|i| InputFeature {
            positives: global.positives,
            negatives: global.negatives,
            color: i.try_into().unwrap(),
            feature_idx: (i + num_nonnull_features).try_into().unwrap(),
        }));

        for i in 0..num_nonnull_features {
            let i = i as usize;
            let c = (colors[i] as usize) + num_nonnull_features;
            debug_assert!(
                features[i].positives <= features[c].positives,
                "positives double-counted for color {} feature {} stats {:?}",
                c,
                i,
                features[i]
            );
            debug_assert!(
                features[i].negatives <= features[c].negatives,
                "negatives double-counted for color {} feature {} stats {:?}",
                c,
                i,
                features[i]
            );
            features[c].positives -= features[i].positives;
            features[c].negatives -= features[i].negatives;
        }

        // ignore anything that doesn't show up
        features.retain(|feat| feat.count() != 0);

        println!(
            "collect input feature stats {:.0?}",
            Instant::now().duration_since(start)
        );

        // sort everything into independent color columns
        let start = Instant::now();
        features.par_sort_unstable_by(|x, y| {
            (x.color, x.negatives * y.count()).cmp(&(y.color, y.negatives * x.count()))
        });
        println!("sort features {:.0?}", Instant::now().duration_since(start));

        // get accumulate according to each color's sort order
        let start = Instant::now();
        let mut color_offsets = Vec::with_capacity(ncolors + 1);
        let mut cum_features = Vec::with_capacity(features.len() + ncolors);
        let mut current_color: u32 = (ncolors as u32) + 1;
        let mut cumulation = CumulativeFeature::default();
        for feat in features {
            if feat.color != current_color {
                current_color = feat.color;
                cumulation = CumulativeFeature::default();
                cumulation.feature_idx = std::u32::MAX;
                color_offsets.push(cum_features.len());
                cum_features.push(cumulation.clone());
            }

            cumulation.cumulative_negatives += feat.negatives;
            cumulation.cumulative_count += feat.count();
            cumulation.feature_idx = feat.feature_idx;
            cum_features.push(cumulation.clone())
        }
        color_offsets.push(cum_features.len());
        println!("accumulate {:.0?}", Instant::now().duration_since(start));
        assert!(current_color + 1 == ncolors as u32);
        assert!(color_offsets.len() == ncolors + 1);

        // create btrees which contain encoding intervals
        // for every s1, ..., si, ..., sn in a given btree b for a color
        // we'll encode the x-th feature according to that color's ordering
        // in the cum_features vector as i iff s(i-1) < x <= si
        let mut btrees = vec![BTreeSet::default(); ncolors];
        for c in 0..ncolors {
            btrees[c].insert(0);
            btrees[c].insert(color_offsets[c + 1] - color_offsets[c] - 1);
        }

        (
            MutualInformation {
                nfeatures: num_nonnull_features,
                ncolors: ncolors,
                cfc: cum_features,
                co: color_offsets,
                global,
            },
            btrees,
        )
    }

    /// If this feature is a placeholder for 'the k-th color is null', returns Some(k)
    /// else returns None.
    fn null(&self, i: usize) -> Option<u32> {
        let num_nonnull_features = self.nfeatures.try_into().unwrap();
        if self.cfc[i].feature_idx >= num_nonnull_features {
            Some(self.cfc[i].feature_idx - num_nonnull_features)
        } else {
            None
        }
    }

    fn split(&self, di: &mut BTreeSet<usize>, c: usize, i: usize) {
        // 0 and size-1 already added
        assert!(
            self.co[c] < i && i < self.co[c + 1] - 1,
            "lo {} < i {} < hi {} - 1",
            self.co[c],
            i,
            self.co[c + 1]
        );
        let base = self.co[c];

        di.insert(i - base);
    }

    /// undos 'split'
    fn remove(&self, di: &mut BTreeSet<usize>, c: usize, i: usize) {
        // 0 and size-1 already added
        assert!(
            self.co[c] < i && i < self.co[c + 1] - 1,
            "lo {} < i {} < hi {} - 1",
            self.co[c],
            i,
            self.co[c + 1]
        );
        let base = self.co[c];

        di.remove(&(i - base));
    }

    fn gain(&self, di: &BTreeSet<usize>, c: usize, i: usize) -> f64 {
        // 0 and size-1 already added
        assert!(
            self.co[c] < i && i < self.co[c + 1] - 1,
            "lo {} < i {} < hi {} - 1",
            self.co[c],
            i,
            self.co[c + 1]
        );
        let base = self.co[c];
        let (lo, hi) = self.neighbors(di, c, i - base);
        let (lo, hi) = (lo + base, hi + base);
        assert!(lo != i);
        assert!(hi != i);
        let mid = i;
        let last = self.co[c + 1] - 1;

        let denom = self.cfc[last].cumulative_count;
        // denom is denom
        let p = self.cfc[mid].cumulative_count - self.cfc[lo].cumulative_count;
        // denom is denom
        let q = self.cfc[hi].cumulative_count - self.cfc[mid].cumulative_count;
        // denom of alpha is p
        let alpha = self.cfc[mid].cumulative_negatives - self.cfc[lo].cumulative_negatives;
        // denom of beta is q
        let beta = self.cfc[hi].cumulative_negatives - self.cfc[mid].cumulative_negatives;

        let g = self.cvx(denom, p, alpha) + self.cvx(denom, q, beta)
            - self.cvx(denom, p + q, alpha + beta);
        // can be <0 due to numerical issues
        g.max(0.)
    }

    // returns (p/n)f(g/p) with 0 handling
    // where f(t) = t log (t / P[C=0]) + (1 - t) log ((1 - t) / P[C=1])
    fn cvx(&self, n: usize, p: usize, g: usize) -> f64 {
        assert!(n > 0);
        if p == 0 {
            return 0.;
        }
        let tot = self.global.positives + self.global.negatives;
        let pos = self.global.positives;
        let neg = self.global.negatives;
        assert!(pos < tot);
        assert!(neg < tot);
        let g_log_t_pc0 = if g == 0 {
            0.
        } else {
            let g = g as f64;
            let ln_t = g.ln() - (p as f64).ln();
            let ln_pc0 = (neg as f64).ln() - (tot as f64).ln();
            (ln_t - ln_pc0) * g
        };
        let pmg_log_1mt_pc1 = if g == p {
            0.
        } else {
            let pmg = (p - g) as f64;
            let ln_1mt = pmg.ln() - (p as f64).ln();
            let ln_pc1 = (pos as f64).ln() - (tot as f64).ln();
            (ln_1mt - ln_pc1) * pmg
        };
        (g_log_t_pc0 + pmg_log_1mt_pc1) / (n as f64)
    }

    fn neighbors(&self, di: &BTreeSet<usize>, _c: usize, val: usize) -> (usize, usize) {
        use std::ops::Bound::*;

        // https://stackoverflow.com/a/50341316/1779853
        let mut before = di.range((Unbounded, Excluded(val)));
        let mut after = di.range((Excluded(val), Unbounded));

        (*before.next_back().unwrap(), *after.next().unwrap())
    }
}

/// Returns a dictionary from features -> quantized features,
/// which is a smaller contiguous range of integers that maintains
/// as much mutual information with the label as possible.
///
/// Second tuple item returned is a dictionary whose i-the entry is the quantization
/// of the feature "the i-th color is null" -> quantized features, the same
/// range from before.
///
/// Finally, the total number of quantized features used is reported.
///
/// Note that some features may not appear in the training set and as
/// a result do not have a quantized value.
fn quantize(
    mi: MutualInformation,
    mut di: Vec<BTreeSet<usize>>,
    budget: usize,
    sort: bool,
) -> (Vec<Option<u32>>, Vec<Option<u32>>, usize) {
    assert!(
        mi.ncolors <= budget,
        "need at least as much budget {} as colors {}",
        budget,
        mi.ncolors
    );
    // there's an implicit 'divider' between each color.
    let budget = budget - mi.ncolors + 1;

    let insertions = if sort {
        // same as the not-sort routine, but done on a per-variable basis
        // and then globally sorted by marginal mutual information
        let start = Instant::now();
        let mut heaps = vec![BinaryHeap::new(); mi.ncolors];
        for c in 0..mi.ncolors {
            for i in (mi.co[c] + 1)..(mi.co[c + 1] - 1) {
                heaps[c].push((NotNan::new(mi.gain(&di[c], c, i)).unwrap(), i));
            }
        }
        println!("heap init {:.0?}", Instant::now().duration_since(start));
        println!("heap sz {}", heaps.iter().map(|h| h.len()).sum::<usize>());

        let start = Instant::now();
        let mut all_splits: Vec<_> = di
            .par_iter_mut()
            .zip(heaps.into_par_iter())
            .enumerate()
            .flat_map(|(c, (mut di, mut heap))| {
                let ref mut di = di;
                let mut splits = Vec::new();
                let mut insertions = 0;
                while !heap.is_empty() && insertions <= budget {
                    let (_, i) = heap.pop().unwrap();
                    let gain = mi.gain(di, c, i);
                    if gain == 0. {
                        continue;
                    }
                    let ub = heap.peek().map(|(g, _)| g.into_inner()).unwrap_or(0.);
                    if gain >= ub {
                        insertions += 1;
                        mi.split(di, c, i);
                        splits.push((c, i));
                    } else {
                        heap.push((NotNan::new(mi.gain(di, c, i)).unwrap(), i));
                    }
                }

                splits
                    .into_iter()
                    .map(|(c, i)| {
                        mi.remove(di, c, i);
                        let marginal_gain = mi.gain(di, c, i);
                        mi.split(di, c, i);
                        (NotNan::new(marginal_gain).unwrap(), c, i)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        println!("lazy greedy {:.0?}", Instant::now().duration_since(start));

        // sort stage

        let start = Instant::now();
        all_splits.sort_unstable();
        all_splits.reverse();
        println!(
            "sort by marginal gain {:.0?}",
            Instant::now().duration_since(start)
        );

        let start = Instant::now();
        while all_splits.len() > budget {
            let (_, c, i) = all_splits.pop().unwrap();
            mi.remove(&mut di[c], c, i);
        }
        println!("pop extras {:.0?}", Instant::now().duration_since(start));

        all_splits.len()
    } else {
        let start = Instant::now();
        let mut heap = BinaryHeap::new();
        for c in 0..mi.ncolors {
            for i in (mi.co[c] + 1)..(mi.co[c + 1] - 1) {
                heap.push((NotNan::new(mi.gain(&di[c], c, i)).unwrap(), c, i));
            }
        }
        println!("heap init {:.0?}", Instant::now().duration_since(start));
        println!("heap sz {}", heap.len());

        let start = Instant::now();
        let mut insertions = 0;
        loop {
            // stochastic lazy greedy might be even better
            if heap.is_empty() || insertions == budget {
                break;
            }
            let (c, i) = match heap.pop() {
                None => break,
                Some((_, c, i)) => (c, i),
            };
            let gain = mi.gain(&di[c], c, i);
            if gain == 0. {
                continue;
            }
            let ub = heap.peek().map(|(g, _, _)| g.into_inner()).unwrap_or(0.);
            if gain >= ub {
                insertions += 1;
                mi.split(&mut di[c], c, i);
            } else {
                heap.push((NotNan::new(mi.gain(&di[c], c, i)).unwrap(), c, i));
            }
        }
        println!("lazy greedy {:.0?}", Instant::now().duration_since(start));
        insertions
    };

    println!("used up {} columns of {} budget", insertions, budget);

    let mut quantized = vec![None; mi.nfeatures];
    let mut nulls = vec![None; mi.ncolors];

    let start = Instant::now();
    let mut ctr = 0;
    for c in 0..mi.ncolors {
        let base = mi.co[c];
        let mut prev = 0;
        for i in di[c].iter().skip(1).copied() {
            assert!(prev + 1 <= i);
            for j in (prev + 1)..=i {
                let k = base + j;
                if let Some(color_null) = mi.null(k) {
                    let color_null = color_null as usize;
                    assert!(nulls[color_null].is_none());
                    nulls[color_null] = Some(ctr);
                } else {
                    let feature = mi.cfc[k].feature_idx as usize;
                    assert!(quantized[feature].is_none());
                    quantized[mi.cfc[k].feature_idx as usize] = Some(ctr);
                }
            }
            ctr += 1;
            prev = i;
        }
        assert!(mi.co[c] + prev + 1 == mi.co[c + 1]);
    }
    println!(
        "extract dictionary {:.0?}",
        Instant::now().duration_since(start)
    );
    (quantized, nulls, ctr as usize)
}

pub(crate) fn skip_example(d: DelimIter<'_>, split_rate: usize) -> bool {
    // for split variant
    let mut hasher: XXHasher = Default::default();
    for word in d {
        word.hash(&mut hasher);
        b' '.hash(&mut hasher);
    }
    (hasher.finish() & ((1 << split_rate) - 1)) == 0
}
