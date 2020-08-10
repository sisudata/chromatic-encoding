//! Main file runs the command-line application for `csl`, or chromatic
//! sparse learning.

use std::error::Error;
use std::path::PathBuf;
use structopt::clap::arg_enum;
use structopt::StructOpt;

use csl;

arg_enum! {

#[derive(Debug)]
enum Compression {
    TargetEncode,
    SubmodularExpansion,
    SubmodularExpansionNoSplit,
    FrequencyTruncation,
    FieldAwareFrequencyTruncation,
    SubmodularSort,
    NoSplitSubmodularSort,
    Identity
}

}

/// Reads a set of training and validation set files from disk. Each
/// file is treated as a u8 byte block. The only special characters
/// are '\n', ':', and ' '. The first word on each line should be ASCII
/// numerical. The typical SVMlight [1] format is expected, but any
/// values that sparse features (i.e., features present in less than 10%
/// of rows) take on are stripped (all sparse features are assumed to have value 1).
/// A value is not required (then it is assumed to be 1). I.e.,
/// <feature> == <feature>:1, and this is equal to <feature>:3.212 for
/// sparse features (but the latter is distinct for dense features).
///
/// Note that if the tsv_dense flag is specified, '\t' is the separator instead
/// of ' '.
///
/// Note that <feature> values don't have to be numbers; everything is hashed
/// into 64 bits (so collisions are possible but should be exceedingly rare).
///
/// Note proper formatting is not checked.
///
/// Repeat features are OK but an unnecessary slowdown. If you have this
/// problem, try modifying this SO awk script to pre-process your data [2].
///
/// Writes to adjacent files with a new suffix .te<budget>.svm or
/// .sm<budget>.svm (adjacent, meaning they're placed next to the
/// originating file), where te is used to designate target encoding
/// and sm means submodular optimization.
///
/// Note for te the features will now have some numerical designation,
/// but for sm they will be one-hot still. The budget variables
/// controls the maximum number of generated output columns.
///
/// The sm option, by virtue of creating sparse output still, sticks to the single
/// file (per input file) svm format.
///
/// Calls into rayon, so use `RAYON_NUM_THREADS` env variable to
/// control thread count.
///
/// Finally, note SubmodularExpansion uses half the data for estimating conditional
/// probabilities, and writes out only the other half.
///
/// All methods will write out a *.field_dims.txt file which contains the newline-separated
/// number of features that each output field has. For one-hot or numeric outputs, the
/// number of features is 1. For categorical outputs, the number is equal to the cardinality
/// of the resulting column.
///
/// [1] http://svmlight.joachims.org/
/// [2] https://unix.stackexchange.com/a/358653/269078
#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Training set, to be read in parallel (space separated).
    #[structopt(long)]
    train: Vec<PathBuf>,

    /// Validation set, as a sequence of SVMlight files. Like training,
    /// this is read in parallel as well. The target values of
    /// this dataset are not looked at (they can be junk) (space separated).
    #[structopt(long)]
    valid: Vec<PathBuf>,

    /// If this option is set to a list of values, as opposed to not set at all,
    /// then a big special case occurs. The input gets interpreted as a tsv.
    /// Further, the columns in this list are considered the continuous / dense
    /// columns while other ones are considered categorical.
    ///
    /// No coloring takes place, but feature compression does.
    ///
    /// Output is still going to end up being written in SVM format; that part
    /// is not special cased (but no coloring occurs, since input features are
    /// now expected to be in a dense format for the tsv).
    ///
    /// Ideally, I'd make this a whole different executable, but I don't have
    /// time for that.
    ///
    /// Column indices should start counting after the target (the first word).
    #[structopt(long)]
    tsv_dense: Option<Vec<usize>>,

    /// Designate the compression methodology after performing coloring, which
    /// can be either target encoding or submodular expansion.
    #[structopt(long, possible_values = &Compression::variants(), case_insensitive = true)]
    compress: Compression,

    /// The column budget controls what the maximum number of output columns
    /// will be. For target encoding, this can be relatively small, there's no
    /// sense in it being larger than the chromatic number of the co-ocurrence
    /// graph (printed out during the run). For submodular expansion,
    /// which converts back to one-hot, this should be larger and defined by
    /// feature size constraints.
    #[structopt(long)]
    budget: usize,

    /// Trim out features that are below the cutoff in terms of frequency
    #[structopt(long, default_value = "0")]
    freq_cutoff: usize,

    /// Use 2**(-split_rate) proportion of the dataset to estimate
    /// conditional probabilities.
    #[structopt(long, default_value = "1")]
    split_rate: usize,

    /// If specified, dump the graph in text format (edge on each line,
    /// space between vertices) to this file.
    #[structopt(long)]
    dump_graph: Option<PathBuf>,

    /// If --print-new-edges is specified, then prints diagnostic information
    /// about the frequency of new edges (and new edges between vertices of
    /// the same color)
    #[structopt(long)]
    print_new_edges: bool,

    /// If --print-new-edges is specified, then uses this value of k to show
    /// diagnostics about color collisions for the filtered, thresholded graph
    /// given by threshold k, for varying colors as specified by
    /// the argument diagnostic_colors
    ///
    /// If this is set to 0 (the default), then these diagnostics are not printed.
    #[structopt(long, default_value = "0")]
    k: usize,

    /// If --print-new-edges is specified, then uses this value of max_k
    /// to show graphical diagnostics about the k-filtered co-occurrence
    /// graphs, with k varying between [1, max_k].
    #[structopt(long, default_value = "64")]
    max_k: usize,

    /// See `k` argument. If left empty then a single reasonable number
    /// of colors will be chosen for you for Glauber coloring.
    #[structopt(long)]
    diagnostic_colors: Option<Vec<usize>>,

    /// See `k` argument. If --nofilter is on, then for the diagnostics
    /// printed due to the `k` flag (collisions from Glauber coloring),
    /// no pre-emptive largest-first filtering is performed.
    #[structopt(long)]
    nofilter: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    let (compress, suffix) = match opt.compress {
        Compression::TargetEncode => (
            csl::Compression::TargetEncode,
            format!("{}te.svm", opt.budget),
        ),
        Compression::SubmodularExpansion => (
            csl::Compression::SubmodularExpansion,
            format!("{}sm.svm", opt.budget),
        ),
        Compression::SubmodularExpansionNoSplit => (
            csl::Compression::SubmodularExpansionNoSplit,
            format!("{}sn.svm", opt.budget),
        ),
        Compression::FrequencyTruncation => (
            csl::Compression::FrequencyTruncation,
            format!("{}ft.svm", opt.budget),
        ),
        Compression::FieldAwareFrequencyTruncation => (
            csl::Compression::FieldAwareFrequencyTruncation,
            format!("{}faft.svm", opt.budget),
        ),
        Compression::SubmodularSort => (
            csl::Compression::SubmodularSort,
            format!("{}ss.svm", opt.budget),
        ),
        Compression::NoSplitSubmodularSort => (
            csl::Compression::NoSplitSubmodularSort,
            format!("{}ns.svm", opt.budget),
        ),
        Compression::Identity => (csl::Compression::Identity, format!("{}id.svm", opt.budget)),
    };

    csl::read_featurize_write(
        opt.train,
        opt.valid,
        opt.tsv_dense,
        &suffix,
        compress,
        opt.budget,
        opt.dump_graph,
        opt.freq_cutoff,
        opt.print_new_edges,
        opt.split_rate,
        opt.k,
        opt.max_k,
        opt.diagnostic_colors.unwrap_or(vec![]),
        opt.nofilter,
    )?;
    Ok(())
}
