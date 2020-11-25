//! # `crank` - CHRomatic ENcoding Crate
//!
//! Utilities for reading and writing svmlight-like and CSC-like files.

mod scanner;

pub use scanner::{DelimIter, Scanner};
