//! I had to write this b/c python svmlight parsing is so damn slow
//!
//! If you have x.svm, and svmlight file, then running
//!
//! cargo run svm2bins x.svm
//!
//! Generates x.data.bin, x.indices.bin, x.indptr.bin, and x.y.bin,
//! in CSR format, always as flat f64, u32, u64, f64 arrays,
//! in native endianness. Note if you just hand these of to
//! scipy.sparse.csr_matrix you should check for 1-indexed vs 0-indexed
//! data (usually svmlight is the former).
//!
//! Call an appropriately-typed np.fromfile to load.
use byte_slice_cast::IntoByteVec;
use std::error::Error;
use std::io::Write;
use std::path::PathBuf;

use csl;

fn main() -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("Usage: cargo run svm2bins x.svm"),
    );

    let scan =
        csl::svm_scanner::SvmScanner::new(vec![path.clone()], rayon::current_num_threads(), b' ')?;

    let (data, indices, indptr, y) = scan.fold_reduce_serial(
        || (vec![], vec![], vec![0], vec![]),
        |(mut data, mut indices, mut indptr, mut y), mut word_iter| {
            // SAFETY: caller promised the first word is f64 in ascii
            // in SvmScanner guarantees
            let target: &str =
                unsafe { std::str::from_utf8_unchecked(word_iter.next().expect("target")) };
            let target: f64 = target.parse().expect("float format");
            y.push(target);

            for word in word_iter {
                if !csl::feature::not_space(word) {
                    continue;
                }
                let (feature, value) = csl::feature::pair_value(word);
                let feature: &str = unsafe { std::str::from_utf8_unchecked(feature) };
                let value: &str = unsafe { std::str::from_utf8_unchecked(value) };
                let feature: u32 = feature.parse().expect("u32 format");
                let value: f64 = value.parse().expect("float format");

                indices.push(feature);
                data.push(value);
            }
            indptr.push(data.len() as u64);
            (data, indices, indptr, y)
        },
        join_sps,
    );

    let mut dataf = csl::svm_scanner::replace_extension(&path, "data.bin").1;
    let mut indicesf = csl::svm_scanner::replace_extension(&path, "indices.bin").1;
    let mut indptrf = csl::svm_scanner::replace_extension(&path, "indptr.bin").1;
    let mut yf = csl::svm_scanner::replace_extension(&path, "y.bin").1;

    let data = data.into_byte_vec();
    dataf.write_all(&data).expect("write succeeded for dataf");
    let indices = indices.into_byte_vec();
    indicesf
        .write_all(&indices)
        .expect("write succeeded for indicesf");
    let indptr = indptr.into_byte_vec();
    indptrf
        .write_all(&indptr)
        .expect("write succeeded for indptrf");
    let y = y.into_byte_vec();
    yf.write_all(&y).expect("write succeeded for yf");

    Ok(())
}

fn join_sps(
    mut l: (Vec<f64>, Vec<u32>, Vec<u64>, Vec<f64>),
    mut r: (Vec<f64>, Vec<u32>, Vec<u64>, Vec<f64>),
) -> (Vec<f64>, Vec<u32>, Vec<u64>, Vec<f64>) {
    if l.0.len() < r.0.len() {
        return join_sps(r, l);
    }
    let rstart = l.0.len() as u64;
    r.2.iter_mut().for_each(|i| *i += rstart);
    l.0.extend(r.0.into_iter());
    l.1.extend(r.1.into_iter());
    l.2.extend(r.2.into_iter().skip(1));
    l.3.extend(r.3.into_iter());
    l
}
