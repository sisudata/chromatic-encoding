//! Generates a feature co-occurrence graph from a dataset.

use std::collections::HashMap;
use std::convert::TryInto;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use std::u32;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde_json::json;
use structopt::StructOpt;

use crank::{graphio, simsvm, Scanner};

/// Reads utf-8 text files in simsvm format. "simplified svmlight" format.
///
/// Generates the assocaited feature co-occurrence graph.
/// The graph file is an svmlight file with lines like
///
/// <target> <neightbor>:<edge weight>...
///
/// where a given line contains nodes adjacent to the first <target> node
/// and their edge weights (frequency of occurrence).
/// The file is outputted as ${dataset}.graph all in text format, and each
/// node is only ever a <target> once. Redundant bidirectional edges are not encoded.
#[derive(Debug, StructOpt)]
#[structopt(name = "graph", about = "Generate the feature co-occurrence graph.")]
struct Opt {
    /// Files to build the co-occurrence graph from.
    #[structopt(long)]
    files: Vec<PathBuf>,

    /// Output files for resulting graph.
    #[structopt(long)]
    out: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let scanner = Scanner::new(opt.files, b' ');
    let stats_start = Instant::now();
    let stats = simsvm::stats(&scanner);
    println!(
        "{}",
        json!({
            "dataset": "train",
            "nfeatures": stats.nfeatures(),
            "npages": stats.npages(),
            "nrows": stats.nrows(),
            "avg_nnz": stats.avg_nnz(),
            "stats_duration": format!("{:.0?}", Instant::now().duration_since(stats_start)),
        })
    );

    let csr_start = Instant::now();
    let csr = simsvm::csr(&scanner, &stats);
    println!(
        "{}",
        json!({ "csr_duration": format!("{:.0?}", Instant::now().duration_since(csr_start)) })
    );

    let graph_start = Instant::now();
    let (nedges, avg_edge_weight, avg_degree) = graphio::write(&csr, stats.nfeatures(), &opt.out);
    println!(
        "{}",
        json!({
            "nedges": nedges,
            "nvertices": stats.nfeatures(),
            "avg_edge_weight": avg_edge_weight,
            "avg_degree": avg_degree,
            "graph_duration": format!("{:.0?}", Instant::now().duration_since(graph_start)),
        })
    );
}
