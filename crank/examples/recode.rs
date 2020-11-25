//! Recode mainfile remaps words for a set of files based on an input
//! dictionary.

use std::collections::HashMap;
use std::io;
use std::io::{Read, Write};
use std::path::PathBuf;

use structopt::StructOpt;

use crank::{DelimIter, Scanner};

/// Reads a set of key-value pairs, space-delimited, from stdin.
/// Expects a list of file names as input. Files should have utf-8 text
/// and have each start with a word (not a space or empty line).
/// Creates adjacent files `<original_filename>_recode` with all words
/// after the first re-mapped according to the key-value pairs, or
/// empty string if no key matches.
#[derive(Debug, StructOpt)]
#[structopt(name = "recode", about = "Recoding a list of files with map.")]
struct Opt {
    /// Files to recode.
    #[structopt(long)]
    files: Vec<PathBuf>,
}

fn main() {
    let opt = Opt::from_args();
    let mut buf = Vec::with_capacity(64 * 1024);
    io::stdin().lock().read_to_end(&mut buf).unwrap();
    let kv: HashMap<&[u8], &[u8]> = DelimIter::new(&buf, b'\n')
        .map(|line| {
            let mut iter = DelimIter::new(line, b' ');
            let key = iter.next().expect("key");
            let value = iter.next().expect("value");
            (key, value)
        })
        .collect();
    let scanner = Scanner::new(opt.files, b' ');
    scanner.for_each_sink(
        (),
        |mut line, writer, _| {
            writer.write_all(line.next().expect("first key")).unwrap();
            line.for_each(|word| {
                writer.write_all(b" ").unwrap();
                kv.get(word)
                    .iter()
                    .for_each(|&v| writer.write_all(v).unwrap())
            });
            writer.write_all(b"\n").unwrap();
        },
        "_recode",
    );
}
