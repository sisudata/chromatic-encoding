//! This module helps us efficiently read from sequences of SVMlight [1] files
//!
//! In particular, the only special characters expected to live in these files are
//! ' ' and '\n'.
//!
//! Each line is expected to look like:
//!
//! <target> <feature> <feature> <feature>...
//!
//! So the usual feature/value pair specified in SVMlight as <feature>:<value>
//! is interpreted as a feature in itself (with distinct values being represented
//! as unique features). This works as expected for categorical features, but not
//! so for numerical, so please extract numerical features out before processing.
//!
//! This module also provides "parallel writing" capabilities, which allow the user
//! to create files in parallel that contain the same number of lines as the original
//! input to the scanner.
//!
//! The format is very restrictive.
//! As far as this module is concerned, each line consists of space-separated words.
//! The users of this will typically depend on the facts that:
//!
//!   1. Words should have non-empty length (no two adjacent spaces).
//!   2. Lines always contain a numerical target encoded in ASCII, e.g., "3.14"
//!   3. There may be zero or more features.
//!
//! [1] http://svmlight.joachims.org/

use bstr::ByteSlice;
use itertools::Itertools;
use memmap::Mmap;
use memmap::MmapOptions;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::iter;
use std::path::PathBuf;
use std::process::Command;
use std::result::Result;

/// Stores which mmap this byte block refers to, its offset within that mmap,
/// and its length.
#[derive(Debug, Clone)]
struct ByteBlock {
    mmap_index: usize,
    mmap_offset: usize,
    length: usize,
}

/// An iterator over byte slices separated by a delimiter.
/// The iterated-over slices won't contain the delimiter, but may be empty.
#[derive(Clone)]
pub struct DelimIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    delim: u8,
}

impl<'a> DelimIter<'a> {
    fn new<'b>(bytes: &'b [u8], delim: u8) -> DelimIter<'b> {
        DelimIter {
            bytes,
            pos: 0,
            delim,
        }
    }

    /// Assuming contents are utf8, returns them.
    pub(crate) fn dbg_line(&self) -> String {
        let clone = self.clone();
        clone
            .map(|w| format!("{}", std::str::from_utf8(w).unwrap_or("<BAD-UTF8>")))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl<'a> Iterator for DelimIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.pos == self.bytes.len() {
            None
        } else {
            let start = self.pos;
            let bytes = &self.bytes[start..];
            let (end, new_pos) = match bytes.find_byte(self.delim) {
                None => (bytes.len(), bytes.len()),
                Some(next_line) => (next_line, next_line + 1),
            };
            self.pos = start + new_pos;
            Some(&bytes[..end])
        }
    }
}

/// An SvmScanner provides efficient line-level access to underlying files.
///
/// This is done through mmapping, so the user must promise the files won't
/// be concurrently modified for the duration of use.
///
/// Internally, an SvmScanner divides its input files into a set of file-aligned
/// byte blocks, which are the unit of parallelism used in processing.
///
/// Please note that the format is very restrictive. See the module documentation
/// above.
pub struct SvmScanner {
    paths: Vec<PathBuf>,
    mmaps: Vec<Mmap>,
    blocks: Vec<ByteBlock>,
    delimiter: u8,
    nthreads: usize,
}

impl SvmScanner {
    pub(crate) fn nthreads(&self) -> usize {
        self.nthreads
    }

    /// `nthreads` is the desired parallelism level for the processing code
    /// (and should align with the default `rayon` pool used during processing).
    ///
    /// Internally, this parameter affects the number of blocks used.
    pub fn new(
        paths: Vec<PathBuf>,
        nthreads: usize,
        delimiter: u8,
    ) -> Result<SvmScanner, Box<dyn Error>> {
        // TODO: panic if repeat path

        // Blocks determine the number of atomic chunks to divide inputs into.
        // To minimize the number of merges, this should not be significantly
        // more than nthreads. To minimize wasted time from work imbalance,
        // this should be a little larger than `nthreads`. Note that the user
        let nblocks = nthreads * 4;

        // Consider reading whole thing into memory with ByteSlice::from_path.
        let mmaps = paths
            .iter()
            .map(|path: &PathBuf| -> Result<_, Box<dyn Error>> {
                let file = File::open(path)?;
                // SAFETY: caller must guarantee this file is not
                // mutated during use, per method documentation.
                // https://users.rust-lang.org/t/how-unsafe-is-mmap/19635/24
                Ok(unsafe { MmapOptions::new().map(&file)? })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let file_lengths: Vec<_> = mmaps.iter().map(|mmap| mmap.len()).collect();
        let tot_bytes = file_lengths.iter().sum();
        let nblocks = nblocks.min(tot_bytes / 1024 / 1024).max(1);

        // Compute block start offsets by starting with evenly spaced offsets,
        // referring to a global offset of bytes along the entire list of files.
        let mut offsets: Vec<_> = (0..nblocks).map(|b| b * tot_bytes / nblocks).collect();

        // To make sure we respect file boundaries, throw in all the file offsets.
        let file_offsets: Vec<_> = iter::once(0)
            .chain(file_lengths.iter().scan(0, |state, &flen| {
                *state += flen;
                Some(*state)
            }))
            .collect();
        offsets.extend(file_offsets.iter());
        offsets.sort_unstable();

        // drop the ending global offset (multiple if last few files are empty)
        if let Some(pos) = offsets.iter().position(|&o| o == tot_bytes) {
            offsets.truncate(pos);
        }

        // Compute global offsets, local file offsets, and the file (mmap) index
        // for each proposed offset.
        //
        // Shifts offsets so they point to the character directly after the
        // preceding newline or beginning-of-file, whichever is closer.
        let (global, local_file): (Vec<usize>, Vec<(usize, usize)>) = offsets
            .into_iter()
            .scan(0, |mmap_index, global_offset| {
                if global_offset >= file_offsets[*mmap_index + 1] {
                    *mmap_index += 1;
                };
                let local_offset = global_offset - file_offsets[*mmap_index];
                let shifted_local = start_of_line(local_offset, &mmaps[*mmap_index]);
                let global_offset = global_offset + shifted_local - local_offset;
                Some((global_offset, (shifted_local, *mmap_index)))
            })
            .unzip();

        // Figure out block lengths based on global offsets.
        let block_lengths_rev: Vec<_> = global
            .into_iter()
            .rev()
            .scan(tot_bytes, |state: &mut usize, offset| {
                let blen = *state - offset;
                *state -= blen;
                Some(blen)
            })
            .collect();

        let blocks: Vec<_> = block_lengths_rev
            .into_iter()
            .rev()
            .zip(local_file.into_iter())
            .flat_map(|(blen, (local_offset, mmap_index))| {
                if blen == 0 {
                    None
                } else {
                    Some(ByteBlock {
                        mmap_index,
                        mmap_offset: local_offset,
                        length: blen,
                    })
                }
            })
            .collect();

        assert!(nthreads > 1); // mpsc strategy doesn't work otherwise
        Ok(SvmScanner {
            paths,
            mmaps,
            blocks,
            delimiter,
            nthreads,
        })
    }

    /// Fold over the lines in the associated files to this SvmScanner and combine the
    /// results.
    ///
    /// A (cloneable) one-pass iterator is provided over each line per `fold` invocation.
    pub(crate) fn fold_reduce<'a, U, Id, Fold, Reduce>(
        &'a self,
        ref id: Id,
        fold: Fold,
        reduce: Reduce,
    ) -> U
    where
        U: Send,
        Id: Fn() -> U + Sync + Send,
        Fold: Fn(U, DelimIter<'a>) -> U + Sync + Send,
        Reduce: Fn(U, U) -> U + Sync + Send,
    {
        self.blocks
            .par_iter()
            .map(|block| {
                DelimIter::new(block.bytes(&self), b'\n').fold(id(), |acc, line| {
                    fold(acc, DelimIter::new(line, self.delimiter))
                })
            })
            .reduce(id, reduce)
    }

    /// reduce is completely serial here
    pub fn fold_reduce_serial<'a, U, Id, Fold, Reduce>(
        &'a self,
        ref id: Id,
        fold: Fold,
        reduce: Reduce,
    ) -> U
    where
        U: Send,
        Id: Fn() -> U + Sync + Send,
        Fold: Fn(U, DelimIter<'a>) -> U + Sync + Send,
        Reduce: Fn(U, U) -> U + Sync + Send,
    {
        self.blocks
            .par_iter()
            .map(|block| {
                DelimIter::new(block.bytes(&self), b'\n').fold(id(), |acc, line| {
                    fold(acc, DelimIter::new(line, self.delimiter))
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(id(), reduce)
    }

    /// Loop over each line in parallel; rely on function side effects and shared,
    /// synchronized state to extract information.
    ///
    /// A (cloneable) one-pass iterator is provided over each line per `apply` invocation.
    ///
    /// Finish consumes final state.
    pub(crate) fn for_each<'a, U, Id, Apply, Finish>(
        &'a self,
        ref id: Id,
        apply: Apply,
        finish: Finish,
    ) where
        U: Send,
        Id: Fn() -> U + Sync + Send,
        Apply: Fn(&mut U, DelimIter<'a>) + Sync + Send,
        Finish: Fn(&mut U) + Sync + Send,
    {
        self.blocks.par_iter().for_each(|block| {
            let mut state = id();
            for line in DelimIter::new(block.bytes(&self), b'\n') {
                apply(&mut state, DelimIter::new(line, self.delimiter));
            }
            finish(&mut state)
        });
    }

    /// Map over lines in the associated files, writing to a sink for each file.
    ///
    /// A (cloneable) one-pass iterator is provided over each line's words
    /// is passed per `apply` invocation. You should write out just the contents
    /// and any newlines you'd like to add yourself.
    ///
    /// Creates a new file, one for each input path in this `SvmScanner`, in the
    /// same directory as the input files, with a different suffix. I.e., if we
    /// are scanning over files "f1.svm" and "f2.svm" then the output of this
    /// command will be "f1.<suffix>" and "f2.<suffix>".
    ///
    /// It's on the user to make sure this doesn't overwrite existing files.
    /// If an input file has an extension and it's equivalent to the suffix, shit
    /// will break.
    ///
    /// The `scratch` argument can be used to generate mutable scratch state
    /// that is passed to the apply function.
    pub(crate) fn for_each_sink<Apply, Scratch, T>(
        &self,
        apply: Apply,
        suffix: &str,
        scratch: Scratch,
    ) where
        Apply: Fn(DelimIter<'_>, &mut BufWriter<File>, &mut T) + Send + Sync,
        Scratch: Fn() -> T + Sync + Send,
    {
        // This works by writing out a file for each byte block, then cat-ing them.
        //
        // Suppose we have input files f1.svm and f2.svm
        //
        // We might have byte blocks 0, 1, 2 associated with f1 and 3, 4 with f2.
        // So we'll first create files f1.<suffix>.0 f1.<suffix>.1 f1.<suffix>.2
        // and f2.<suffix>.3 f2.<suffix>.4 in parallel.
        let pad_width = format!("{}", (self.blocks.len() - 1).max(0)).len();
        let grouped_blocks = self
            .blocks
            .iter()
            .enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            // for each block, write it out to its file
            .map(|(i, block)| {
                let block_suffix = format!("{}.{:0width$}", suffix, i, width = pad_width);
                let (block_path, block_file) =
                    replace_extension(&self.paths[block.mmap_index], &block_suffix);
                // 64KB buffer chosen experimentally based on my laptop
                let mut writer = BufWriter::with_capacity(64 * 1024, block_file);
                let mut scratch_state = scratch();
                for line in DelimIter::new(block.bytes(&self), b'\n') {
                    apply(
                        DelimIter::new(line, self.delimiter),
                        &mut writer,
                        &mut scratch_state,
                    );
                }
                writer.flush().unwrap();
                (block.mmap_index, block_path)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .group_by(|(mmap_index, _)| *mmap_index)
            .into_iter()
            .map(|(mmap_index, gby)| (mmap_index, gby.map(|(_, path)| path).collect::<Vec<_>>()))
            .collect::<Vec<_>>();

        grouped_blocks
            .into_par_iter()
            .for_each(|(mmap_index, block_files)| {
                let (_, out_file) = replace_extension(&self.paths[mmap_index], suffix);
                assert!(Command::new("cat")
                    .args(&block_files)
                    .stdout(out_file)
                    .status()
                    .expect("cat failure")
                    .success());
                assert!(Command::new("rm")
                    .args(&block_files)
                    .status()
                    .expect("rm failure")
                    .success());
            });
    }
}

fn start_of_line(buflen: usize, buf: &[u8]) -> usize {
    buf[..buflen]
        .rfind_byte(b'\n')
        .map(|ix| ix + 1)
        .unwrap_or(0)
}

pub fn replace_extension(path: &PathBuf, new_ext: &str) -> (PathBuf, File) {
    let stem = path
        .file_stem()
        .expect("file stem")
        .to_str()
        .expect("valid utf-8 path");
    let new_path = path.with_file_name(format!("{}.{}", stem, new_ext));
    if new_path.exists() {
        println!("out file {:?} exists, will overwrite", new_path);
    };
    let file = File::create(&new_path).unwrap_or_else(|e| panic!("create {:?}: {:?}", new_path, e));
    (new_path, file)
}

impl ByteBlock {
    fn bytes<'a>(&self, scanner: &'a SvmScanner) -> &'a [u8] {
        let start = self.mmap_offset;
        let stop = start + self.length;
        &scanner.mmaps[self.mmap_index][start..stop]
    }
}
