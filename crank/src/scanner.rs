//! This module helps us efficiently read from sequences of text files
//! containing words sepearated by a common delimiter, line-by-line,
//! and in parallel.
//!
//! The chief advantage of this over unix utilities is that it
//! can refered to shared structures in common memory between
//! processing threads.

use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;

use bstr::ByteSlice;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

const BUFSIZE: usize = 64 * 1024;

/// An iterator over byte slices separated by a delimiter.
/// The iterated-over slices won't contain the delimiter, but may be empty.
#[derive(Clone)]
pub struct DelimIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    delim: u8,
}

impl<'a> DelimIter<'a> {
    pub fn new<'b>(bytes: &'b [u8], delim: u8) -> DelimIter<'b> {
        DelimIter {
            bytes,
            pos: 0,
            delim,
        }
    }

    /// Assuming contents are utf8, returns them.
    #[allow(dead_code)]
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

/// A `Scanner` provides efficient line-level access to underlying files of
/// words, where words are delimited with a specified delimiter.
///
/// Outside of that, you're on your own. This means lines that start
/// with the delimiter or have repeat delimiters will have empty words
/// being iterated over.
pub struct Scanner {
    paths: Vec<PathBuf>,
    delimiter: u8,
}

impl Scanner {
    pub fn new(paths: Vec<PathBuf>, delimiter: u8) -> Self {
        // TODO: panic if repeat path
        Self { paths, delimiter }
    }
    /*
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
     */
    /*
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
    }*/

    /// Map over lines in the associated files, writing to a sink for each file.
    ///
    /// A (cloneable) one-pass iterator is provided over each line's words
    /// is passed per `apply` invocation. You should write out just the contents
    /// and any newlines you'd like to add yourself.
    ///
    /// Creates a new file, one for each input path in this `SvmScanner`, in the
    /// same directory as the input files, with an additional suffix. I.e., if we
    /// are scanning over files "f1.svm" and "f2.svm" then the output of this
    /// command will be "f1.svm<suffix>" and "f2.svm<suffix>".
    ///
    /// Common aggregation state is folded over for each file
    pub fn for_each_sink<Apply, T>(&self, init: T, apply: Apply, suffix: &str)
    where
        Apply: Fn(DelimIter<'_>, &mut BufWriter<File>, &mut T) + Send + Sync,
        T: Clone + Send + Sync,
    {
        self.paths.par_iter().for_each(|path| {
            let file = File::open(path).expect("read file");
            let reader = BufReader::with_capacity(BUFSIZE, file);
            let mut fname = path.file_name().expect("file name").to_owned();
            fname.push(&suffix);
            let new_path = path.with_file_name(fname);
            let file = File::create(&new_path).expect("write file");
            let mut writer = BufWriter::with_capacity(BUFSIZE, file);

            let mut agg = init.clone();
            for line in reader.split(b'\n') {
                let line = line.expect("line read");
                apply(DelimIter::new(&line, self.delimiter), &mut writer, &mut agg);
            }
            writer.flush().expect("for each sink flush");
        })
    }
}
