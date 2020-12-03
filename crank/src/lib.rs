//! # `crank` - CHRomatic ENcoding Crate
//!
//! Utilities for reading and writing svmlight-like and CSC-like files.

use sprs::CsMatI;

pub mod color;
mod scanner;
pub mod simsvm;

pub use scanner::{DelimIter, Scanner};

pub type SparseMatrix = CsMatI<(), u32, u32>;
