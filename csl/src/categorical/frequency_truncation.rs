//! Frequency truncation encoder.

use crate::feature::Featurizer;
use std::convert::TryInto;
use std::io::Write;
use std::time::Instant;

pub(crate) struct EncodingDictionary {
    // 'largest' sparse feature
    feature_cutoff: u32,
}

pub(crate) struct Encoder {
    indices: Vec<u32>,
    appeared: Vec<bool>,
}

pub(crate) struct FieldAwareDictionary {
    ncolors: usize,
    colors: Vec<u32>, // only 'feature_cutoff' long
    recode: Vec<u32>, // 1-indexed recoding for the categoricals
    field_dims: Vec<u32>,
}

pub(crate) struct FieldAwareEncoder {
    // 0 is sentinel. See identity.rs sibling.
    color_field: Vec<u32>,
}

fn print_sparse_logs(featurizer: &Featurizer, budget: usize) {
    println!("budget {} for sparse features after dense", budget);
    println!(
        "feature count cutoff {}",
        featurizer.feature_count(
            budget
                .saturating_sub(1)
                .min(featurizer.nsparse() - 1)
                .try_into()
                .unwrap()
        )
    );
}

impl EncodingDictionary {
    pub(crate) fn new(budget: usize, featurizer: &Featurizer) -> Self {
        print_sparse_logs(featurizer, budget);
        EncodingDictionary {
            feature_cutoff: budget.try_into().unwrap(),
        }
    }
}

impl Encoder {
    pub(crate) fn new(dictionary: &EncodingDictionary) -> Self {
        Encoder {
            indices: Vec::with_capacity(dictionary.feature_cutoff as usize),
            appeared: vec![false; dictionary.feature_cutoff as usize],
        }
    }

    pub(crate) fn observe(&mut self, dictionary: &EncodingDictionary, feature: u32) {
        if feature < dictionary.feature_cutoff && !self.appeared[feature as usize] {
            self.indices.push(feature);
            self.appeared[feature as usize] = true;
        }
    }

    pub(crate) fn dense_offset(&self) -> usize {
        self.appeared.len()
    }

    pub(crate) fn finish<W: Write>(&mut self, writer: &mut W) {
        for &i in self.indices.iter() {
            write!(writer, " {}:1", i).expect("successful write");
            self.appeared[i as usize] = false;
        }
        self.indices.clear();
    }
}

impl FieldAwareDictionary {
    pub(crate) fn new(
        budget: usize,
        featurizer: &Featurizer,
        ncolors: usize,
        mut colors: Vec<u32>,
    ) -> Self {
        print_sparse_logs(featurizer, budget);
        assert!(budget >= ncolors, "budget {} < ncolors {}", budget, ncolors);

        colors.truncate(budget);

        let start = Instant::now();
        let mut recode = colors.clone();
        let mut color_counters = vec![1u32; ncolors];
        for r in recode.iter_mut() {
            let color = *r as usize;
            *r = color_counters[*r as usize];
            color_counters[color] += 1;
        }
        println!(
            "recoding categoricals {:.0?}",
            Instant::now().duration_since(start)
        );

        let ncolors = (*colors.iter().max().unwrap() as usize) + 1;

        Self {
            ncolors,
            colors,
            recode,
            field_dims: color_counters,
        }
    }

    /// returns field dims for sparse values only
    pub(crate) fn field_dims(&self) -> Vec<u32> {
        self.field_dims.clone()
    }
}

impl FieldAwareEncoder {
    pub(crate) fn new(dictionary: &FieldAwareDictionary) -> Self {
        Self {
            color_field: vec![0u32; dictionary.ncolors],
        }
    }

    pub(crate) fn observe(&mut self, dictionary: &FieldAwareDictionary, feature: u32) {
        let feature = feature as usize;
        if feature < dictionary.colors.len() {
            let color = dictionary.colors[feature as usize] as usize;
            // 1-index field features for svmlight compat format
            // less popular feature wins
            self.color_field[color] = self.color_field[color].max(dictionary.recode[feature]);
        }
    }

    pub(crate) fn dense_offset(&self) -> usize {
        self.color_field.len()
    }

    pub(crate) fn finish<W: Write>(&mut self, writer: &mut W) {
        for (c, v) in self.color_field.iter_mut().enumerate() {
            if *v == 0 {
                continue;
            }
            write!(writer, " {}:{}", c, v).expect("successful write");
            *v = 0;
        }
    }
}
