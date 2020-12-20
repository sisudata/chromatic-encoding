//! I had to write this b/c python svmlight parsing is so damn slow.
//! Converts svmlight files into a binary encoding of a CSR matrix.

use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use std::u32;

use byte_slice_cast::AsByteSlice;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde_json::json;
use structopt::StructOpt;

use crank::{color, svmlight, Scanner, SummaryStats};

/// Reads utf-8 text files in strict svmlight format, i.e., <target> <feature>:<value>...
///
/// Creates files in the same directory for train and test with
/// a binary encoding of the corresponding CSR matrix.
///
/// I.e., for train, we'll create `<train_out>.{data,indices,indptr,y}`
/// which are all native u32 arrays under native endianness (so feature values
/// are assumed integral and small enough). Except indptr, which is u64.
///
/// Loading these with `np.fromfile` will give `nnz`-long arrays for data and
/// indices files, `nrows`-long `y` array (of 0 or 1), and `nrows + 1` indptr
/// array.
#[derive(Debug, StructOpt)]
#[structopt(name = "svm2bins", about = "svmlight to binary CSR")]
struct Opt {
    /// Training files in svmlight form
    #[structopt(long)]
    train: Vec<PathBuf>,

    /// Base path for outfiles for binary training matrix.
    #[structopt(long)]
    train_out: PathBuf,

    /// Test files to transform.
    #[structopt(long)]
    test: Vec<PathBuf>,

    /// Base path for outfiles for binary training matrix.
    #[structopt(long)]
    test_out: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    for (path, files) in vec![(opt.train_out, opt.train), (opt.test_out, opt.test)] {
        let scanner = Scanner::new(files, b' ');
        let csr = svmlight::csr(&scanner);

        let mut dataf = replace_extension(&path, "data");
        let mut indicesf = replace_extension(&path, "indices");
        let mut indptrf = replace_extension(&path, "indptr");
        let mut yf = replace_extension(&path, "y");

        let data = csr.data.as_byte_slice();
        dataf.write_all(&data).expect("write succeeded for dataf");
        let indices = csr.indices.as_byte_slice();
        indicesf
            .write_all(&indices)
            .expect("write succeeded for indicesf");
        let indptr = csr.indptr.as_byte_slice();
        indptrf
            .write_all(&indptr)
            .expect("write succeeded for indptrf");
        let y = csr.y.as_byte_slice();
        yf.write_all(&y).expect("write succeeded for yf");
    }

    Ok(())
}

fn replace_extension(path: &PathBuf, new_ext: &str) -> File {
    let stem = path
        .file_name()
        .expect("file name")
        .to_str()
        .expect("valid utf-8 path");
    let new_path = path.with_file_name(format!("{}.{}", stem, new_ext));
    if new_path.exists() {
        println!("out file {:?} exists, will overwrite", new_path);
    };
    File::create(&new_path).unwrap_or_else(|e| panic!("create {:?}: {:?}", new_path, e))
}
