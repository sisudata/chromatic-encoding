//! Mean target encoder.

use crate::categorical::sketch;
use crate::feature::Featurizer;
use crate::svm_scanner::SvmScanner;
use std::io::Write;

#[derive(Default, Clone)]
struct Sketch {
    mean: f64,
    count: usize,
}

pub(crate) struct EncodingDictionary {
    features: Vec<Sketch>,
    ncols: usize,
    colors: Vec<u32>,
}

pub(crate) struct Encoder {
    // target encoding value of the i-th color feature
    values: Vec<f64>,
    // count of the feature used to encode the i-th color feature
    counts: Vec<usize>,
    // is this feature set in the current encoding?
    occupied: Vec<bool>,
}

impl sketch::Sketch for Sketch {
    fn update(&mut self, target: f64) {
        self.count += 1;
        self.mean += (target - self.mean) / (self.count as f64);
    }

    fn merge(&mut self, other: &Self) {
        let s = self.count + other.count;
        if s == 0 {
            return;
        }

        let s = s as f64;
        let l = self.count as f64 / s;
        let r = other.count as f64 / s;
        self.mean = self.mean * l + other.mean * r;
        self.count += other.count;
    }
}

impl EncodingDictionary {
    pub(crate) fn new(
        budget: usize,
        train: &SvmScanner,
        featurizer: &Featurizer,
        ncolors: usize,
        colors: Vec<u32>,
    ) -> Self {
        let (_, features) = sketch::map_reduce_sketch::<Sketch, _>(train, featurizer, |_| true);

        println!(
            "stats for counts: {}",
            sketch::pretty_stats(
                features
                    .iter()
                    .map(|sketch| sketch.count)
                    .filter(|&x| x > 0)
                    .collect()
            )
        );

        println!(
            "budget {} {} ncolors {}, will {}",
            budget,
            if budget < ncolors { "<" } else { ">=" },
            ncolors,
            if budget < ncolors {
                "truncate"
            } else {
                "have excess"
            }
        );

        EncodingDictionary {
            colors,
            features: features,
            ncols: ncolors.min(budget),
        }
    }
}

impl Encoder {
    pub(crate) fn new(dictionary: &EncodingDictionary) -> Self {
        let width = dictionary.ncols;
        Encoder {
            values: vec![0f64; width],
            occupied: vec![false; width],
            counts: vec![0usize; width],
        }
    }

    pub(crate) fn observe(&mut self, dictionary: &EncodingDictionary, feature: u32) {
        let h = feature as usize;
        let c = dictionary.colors[h] as usize;

        if c >= self.values.len() {
            return;
        }

        let x = dictionary.features[h].mean;
        let f = dictionary.features[h].count;

        // Features can overlap, but we still want to be deterministic
        // in our outputs, regardless of feature order.

        // Favor less popular features
        if self.occupied[c] && self.counts[c] < f {
            return;
        }

        // Favor smaller values by magnitude
        if self.occupied[c] && self.counts[c] == f && self.values[c].abs() < x.abs() {
            return;
        }

        self.occupied[c] = true;
        self.values[c] = x;
        self.counts[c] = f;
    }

    pub(crate) fn dense_offset(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn finish<W: Write>(&mut self, writer: &mut W) {
        for i in 0..self.counts.len() {
            let x: f64 = self.values[i];

            self.occupied[i] = false;
            self.counts[i] = 0;
            self.values[i] = 0.0;

            if x != 0.0 {
                write!(writer, " {}:{}", i, x).expect("successful write");
            }
        }
    }
}
