//! The identity encoder which keeps the original cateogrical value.

use std::convert::TryInto;
use std::io::Write;
use std::time::Instant;

pub(crate) struct EncodingDictionary {
    ncolors: usize,
    colors: Vec<u32>,
    // map each color's original features into a contiguous range for each index
    // range is 1-indexed, [1, num features in this color]
    recode: Vec<u32>,
    field_dims: Vec<u32>,
}

pub(crate) struct Encoder {
    // The feature that occupies the current color categorical column.
    // sentinel value of 0 means not occupied
    features: Vec<i64>,
    unbiased: bool, // how to handle collisions
}

impl EncodingDictionary {
    pub(crate) fn new(budget: usize, ncolors: usize, colors: Vec<u32>) -> Self {
        assert!(
            budget >= ncolors,
            "budget {} <= ncolors {}",
            budget,
            ncolors
        );

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

        EncodingDictionary {
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

impl Encoder {
    pub(crate) fn new(dictionary: &EncodingDictionary, unbiased: bool) -> Self {
        Encoder {
            features: vec![0i64; dictionary.ncolors],
            unbiased,
        }
    }

    pub(crate) fn observe(&mut self, dictionary: &EncodingDictionary, feature: u32) {
        let h = feature as usize;
        let c = dictionary.colors[h] as usize;
        let code = dictionary.recode[h];

        // Features can overlap if the data is outside the set used for training.

        if self.unbiased {
            // Use a random bit
            self.features[c] += ((h & 1) as i64) * 2 - 1;
        } else {
            // Favor less popular features, i.e., ones with a higher index
            // The way we re-coded was monotonic wrt the original feature index, which was
            // already reverse-sorted with respect to popularity.
            self.features[c] = self.features[c].max(code.try_into().unwrap())
        }
    }

    pub(crate) fn dense_offset(&self) -> usize {
        self.features.len()
    }

    pub(crate) fn finish<W: Write>(&mut self, writer: &mut W) {
        for i in 0..self.features.len() {
            let value = self.features[i];

            if value != 0 {
                write!(writer, " {}:{}", i, value).expect("successful write");
            }

            self.features[i] = 0;
        }
    }
}
