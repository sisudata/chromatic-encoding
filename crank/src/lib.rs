//! # `crank` - CHRomatic ENcoding Crate
//!
//! Utilities for reading and writing svmlight-like and CSC-like files.

use std::collections::HashMap;

use ordered_float::NotNan;
use sprs::CsMatI;

pub mod color;
mod scanner;
pub mod simsvm;

pub use scanner::{DelimIter, Scanner};

pub type SparseMatrix = CsMatI<(), u32, u32>;

const NSTAT_PERCENTILES: usize = 9;
const STAT_PERCENTILES: [f64; NSTAT_PERCENTILES] =
    [0.0, 0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.99, 1.0];

pub struct SummaryStats {
    mean: f64,
    percentiles: [f64; NSTAT_PERCENTILES],
}

impl SummaryStats {
    pub fn from(it: impl Iterator<Item = f64>) -> Self {
        let mut v: Vec<NotNan<f64>> = it.map(|f| NotNan::new(f).unwrap()).collect();
        v.sort_unstable();
        let mut stats = SummaryStats {
            mean: v.iter().map(|f| f.into_inner()).sum::<f64>() / v.len() as f64,
            percentiles: Default::default(),
        };
        STAT_PERCENTILES
            .iter()
            .copied()
            .map(|f| v[((v.len() - 1) as f64 * f) as usize].into_inner())
            .zip(stats.percentiles.iter_mut())
            .for_each(|(val, p)| *p = val);
        stats
    }

    pub fn to_map(&self) -> HashMap<String, f64> {
        let mut map: HashMap<_, _> = STAT_PERCENTILES
            .iter()
            .map(|f| format!("p{:.3}", f))
            .zip(self.percentiles.iter().copied())
            .collect();
        map.insert("mean".to_string(), self.mean);
        map
    }
}
