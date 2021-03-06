//! `csl` stands for chromatic sparse learning. It's a set of
//! hashing-based methods for using graph coloring to perform
//! efficient dimensionality reduction.
//!
//! This is geared towards supervised machine learning, but can be
//! adapted for other settings as well.

mod adjacency;
mod categorical;
mod edges;
pub mod feature;
mod identity;
mod submodular;
pub mod svm_scanner;
mod unsafe_hasher;

use adjacency::AdjacencyList;
use categorical::{Encoder, EncodingDictionary};
use feature::{Featurizer, FeaturizerConstructor};
use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::iter;
use std::path::PathBuf;
use std::time::Instant;
use svm_scanner::SvmScanner;

pub type Compression = categorical::Compression;

pub fn read_featurize_write(
    train: Vec<PathBuf>,
    valid: Vec<PathBuf>,
    tsv_dense: Option<Vec<usize>>,
    out_suffix: &str,
    compress: Compression,
    budget: usize,
    dump_graph: Option<PathBuf>,
    freq_cutoff: usize,
    print_new_edges: bool,
    split_rate: usize,
    threshold_k: usize,
    max_k: usize,
    diagnostic_colors: Vec<usize>,
    nofilter: bool,
) -> Result<(), Box<dyn Error>> {
    let first_train_path = train[0].clone();

    let nthreads = rayon::current_num_threads();
    println!("num threads {}", nthreads);
    let training_start = Instant::now();

    // Initialize our input scanner.
    let start = Instant::now();
    let delimiter = if tsv_dense.is_some() { b'\t' } else { b' ' };
    let train = SvmScanner::new(train, nthreads, delimiter)?;
    println!(
        "intitialize training file scanner {:.0?}",
        Instant::now().duration_since(start)
    );

    // Collect initial statistics: number of lines in the entire dataset, words per line
    // (with repeats), and (approximate) number of distinct words.
    let start = Instant::now();
    let (nlines, nwords, featurizer) = estimate_cardinalities(&train, tsv_dense, freq_cutoff);
    println!(
        "first cardinality scan {:.0?}",
        Instant::now().duration_since(start)
    );
    println!("num lines {}", nlines);
    println!("num words total {}", nwords);
    println!("dense features {}", featurizer.ndense());
    println!("sparse features {}", featurizer.nsparse());
    println!(
        "input data type {}",
        if featurizer.istsv() { "tsv" } else { "svm" }
    );

    let (color, ncolors) = if featurizer.istsv() {
        featurizer.tsvcolors()
    } else {
        // Hot loop: compute edges for adjacency graph. An edge between two features
        // i1 and i2 is just the concatenation of their bits.
        //
        // For logging purposes, also maximum number of active features among all lines.
        let start = Instant::now();
        let (edges_vec, edge_freqs, edge_stats) = edges::collect_edges(&train, &featurizer);
        println!(
            "edge collection {:.0?}",
            Instant::now().duration_since(start)
        );
        println!(
            "edge frequencies {}",
            categorical::sketch::pretty_stats(edge_freqs.clone())
        );
        println!(
            "edge diagnostics {:.0?}",
            Instant::now().duration_since(start)
        );
        println!("num unique edges {}", edges_vec.len());
        println!("avg degree {}", edges_vec.len() * 2 / featurizer.nsparse());
        edge_stats.print();

        if let Some(path) = &dump_graph {
            let start = Instant::now();
            println!("dumping graph to {:?}", path);
            let file = File::create(path).unwrap();
            let mut writer = BufWriter::new(file);
            for e in &edges_vec {
                writeln!(&mut writer, "{} {}", e.left(), e.right()).unwrap();
            }
            writer.flush().unwrap();
            println!("dump graph {:.0?}", Instant::now().duration_since(start));
        }

        let start = Instant::now();
        let mut graph = AdjacencyList::new(featurizer.nsparse(), edges_vec, edge_freqs);
        println!(
            "adjacency list construction {:.0?}",
            Instant::now().duration_since(start)
        );

        if let Some(mut path) = dump_graph {
            let start = Instant::now();
            path.set_extension("degree.txt");

            println!("dumping degrees to {:?}", path);

            let file = File::create(&path).unwrap();
            let mut writer = BufWriter::new(file);
            for v in 0..graph.nvertices() {
                writeln!(&mut writer, "{}", graph.degree(v as u32)).expect("graph write");
            }
            writer.flush().expect("write flush for degree");

            println!("dump degrees {:.0?}", Instant::now().duration_since(start));
        }

        let start = Instant::now();
        let (ncolors, color) = greedy_color(&graph);
        println!(
            "greedy graph coloring {:.0?}",
            Instant::now().duration_since(start)
        );
        println!("greedy num colors {}", ncolors);

        if print_new_edges {
            let valid = SvmScanner::new(valid.clone(), nthreads, delimiter)?;
            let (edges_vec, edge_freqs, edge_stats) = edges::collect_edges(&valid, &featurizer);
            let mut valid_graph = AdjacencyList::new(featurizer.nsparse(), edges_vec, edge_freqs);
            let valid_n = edge_stats.nlines();

            println!("evaluating graphs at different thresholds k");

            let start = Instant::now();

            graph.internal_sort();
            valid_graph.internal_sort();

            let train_n = nlines;

            let (rate, std) = color_collision_count(&valid, &featurizer, &color);
            println!("color collision count {}", rate);
            println!("color collision std {}", std);

            if max_k > 0 {
                // substitute 3 with 64 for graphs in Appendix
                let ks = (1..=max_k).collect::<Vec<_>>();
                let mut gt_ests = Vec::new();
                let mut actual_new_edges = Vec::new();
                let mut max_degrees = Vec::new();
                let mut avg_degrees = Vec::new();

                let mut good_turing = 0;
                let mask = vec![true; graph.nvertices()];
                for &k in &ks {
                    good_turing += k * graph.appeared_k(k, &mask);
                    let gt_est = good_turing as f64 / train_n as f64;
                    gt_ests.push(gt_est);

                    let missing_edgecount = graph.nmissing(k, &mut valid_graph, &mask);
                    let actual = missing_edgecount as f64 / valid_n as f64;
                    actual_new_edges.push(actual);

                    let (max_degree, avg_degree) = graph.filter_degree(k);
                    max_degrees.push(max_degree);
                    avg_degrees.push(avg_degree);
                }

                println!("thresholded k {:?}", ks);
                println!("thresholded max degree {:?}", max_degrees);
                println!("thresholded avg degree {:?}", avg_degrees);
                println!("thresholded Good-Turing estimate {:?}", gt_ests);
                println!("thresholded actual new edge avg {:?}", actual_new_edges);

                let mut gt_ests = Vec::new();
                let mut actual_new_edges = Vec::new();
                let mut max_degrees = Vec::new();
                let mut avg_degrees = Vec::new();
                let mut discrete_vertices = Vec::new();

                let par_results = ks
                    .par_iter()
                    .map(|&k| {
                        let unfiltered_graph = &graph;

                        let mut graph = graph.filter(k);
                        graph.internal_sort();

                        // We represent the `cutoff`-largest degree vertices of the
                        // largest-first order "discretely", choosing cutoff such that
                        // twice the degree of the subgraph without the discrete vertices
                        // is smaller than the number of discrete vertices. Using fewer than
                        // this many vertices doesn't make sense, since you need to color with
                        // at least 2 * degree colors anyway, generating that many columns.
                        let lf = graph.largest_first();
                        let cutoff = lf
                            .iter()
                            .map(|(_, deg)| *deg as usize)
                            .enumerate()
                            .position(|(i, deg)| deg * 2 < i)
                            .unwrap_or(lf.len());

                        let mut mask = vec![true; lf.len()];
                        for (v, _) in &lf[0..cutoff] {
                            mask[*v as usize] = false;
                        }

                        let good_turing = (1..=k)
                            .map(|j| j * unfiltered_graph.appeared_k(j, &mask))
                            .sum::<usize>();
                        let gt_est = good_turing as f64 / train_n as f64;

                        let missing_edgecount = graph.nmissing(k, &valid_graph, &mask);
                        let actual = missing_edgecount as f64 / valid_n as f64;

                        (
                            gt_est,
                            cutoff,
                            actual,
                            graph.max_degree(&mask),
                            graph.avg_degree(&mask),
                        )
                    })
                    .collect::<Vec<_>>();

                for x in par_results {
                    let (gt_est, cutoff, actual, max_degree, avg_degree) = x;
                    gt_ests.push(gt_est);
                    discrete_vertices.push(cutoff);
                    actual_new_edges.push(actual);
                    max_degrees.push(max_degree);
                    avg_degrees.push(avg_degree);
                }

                println!("filtered thresholded k {:?}", ks);
                println!("filtered thresholded num discrete {:?}", discrete_vertices);
                println!("filtered thresholded max degree {:?}", max_degrees);
                println!("filtered thresholded avg degree {:?}", avg_degrees);
                println!("filtered thresholded Good-Turing estimate {:?}", gt_ests);
                println!(
                    "filtered thresholded actual new edge avg {:?}",
                    actual_new_edges
                );
            }

            if threshold_k > 0 {
                let collisions: Vec<_> = diagnostic_colors
                    .par_iter()
                    .map(|&c| {
                        let mut graph = graph.filter(threshold_k);
                        graph.internal_sort();

                        let mut mask = vec![true; graph.nvertices()];

                        if !nofilter {
                            let lf = graph.largest_first();
                            let cutoff = lf
                                .iter()
                                .map(|(_, deg)| *deg as usize)
                                .enumerate()
                                .position(|(i, deg)| deg * 2 < i)
                                .unwrap_or(lf.len());
                            for (v, _) in &lf[0..cutoff] {
                                mask[*v as usize] = false;
                            }
                        }

                        let (ncolors, colors) = glauber_color(&graph, c as u32, &mask);
                        let (rate, std) = color_collision_count(&valid, &featurizer, &colors);
                        (ncolors, rate, std)
                    })
                    .collect();
                println!("glauber collisions {:?}", collisions);
            }

            println!(
                "new color co-occurrence assessment {:.0?}",
                Instant::now().duration_since(start)
            );
        }

        if cfg!(debug_assertions) {
            validate_coloring(ncolors as usize, &color, &train, &featurizer);
        }
        (color, ncolors)
    };

    // Compute feature counts, which is the number of lines the feature appears in,
    // and the target encoding at the same time.
    let start = Instant::now();
    let encoding_dictionary = EncodingDictionary::new(
        compress,
        budget,
        &train,
        &featurizer,
        ncolors as usize,
        color,
        split_rate,
    );
    println!(
        "categorical encoding {:.0?}",
        Instant::now().duration_since(start)
    );

    let start = Instant::now();
    write_encoding(&encoding_dictionary, &train, &out_suffix, &featurizer, true);
    println!(
        "convert training {:.0?}",
        Instant::now().duration_since(start)
    );
    let e2e = Instant::now().duration_since(training_start);
    println!("e2e pipeline time {:.0?}", e2e);

    let start = Instant::now();
    let valid = SvmScanner::new(valid, nthreads, delimiter)?;
    write_encoding(
        &encoding_dictionary,
        &valid,
        &out_suffix,
        &featurizer,
        false,
    );
    println!("convert valid {:.0?}", Instant::now().duration_since(start));

    let out_suffix = out_suffix.to_owned() + ".field_dims.txt";
    let mut field_dims = encoding_dictionary.field_dims();
    field_dims.extend(iter::repeat(1).take(featurizer.ndense()));
    let (_, mut outf) = svm_scanner::replace_extension(&first_train_path, &out_suffix);
    for f in field_dims {
        writeln!(outf, "{}", f).expect("successful write");
    }

    Ok(())
}

/// Count total number of lines and words referred to by data, as well as
/// number of unique words, up to the specified relative error.
fn estimate_cardinalities(
    train: &SvmScanner,
    tsv_dense: Option<Vec<usize>>,
    freq_cutoff: usize,
) -> (usize, usize, Featurizer) {
    let (nlines, words, unique_words) = train.fold_reduce(
        || {
            (
                0,
                0,
                FeaturizerConstructor::new(freq_cutoff, tsv_dense.clone()),
            )
        },
        |(nlines, mut words, mut unique_words), word_iter| {
            words += unique_words.read_words(word_iter);
            (nlines + 1, words, unique_words)
        },
        |(l_nlines, l_words, mut l_unique_words), (r_nlines, r_words, mut r_unique_words)| {
            l_unique_words.merge(&mut r_unique_words);
            (l_nlines + r_nlines, l_words + r_words, l_unique_words)
        },
    );
    (nlines, words, unique_words.build())
}

/// Intermediate structure used for gathering statistics about a particular feature.
// Q: Why keep an array of structs around vs struct of arrays?
// A: 13s -> 6s on 8 thread Malicious URLs data
#[derive(Clone, Default, Debug)]
struct FeatureStats {
    most_recent_line: usize,
    count: usize,
    target: f64,
}

/// Greedily color the graph with a largest-first strategy.
///
/// Colors are numbered starting at 0, with smaller colors being
/// the ones that contain the largest number of associated vertices.
fn greedy_color(graph: &AdjacencyList) -> (u32, Vec<u32>) {
    let nvertices = graph.nvertices();
    let mut vertices: Vec<_> = (0..nvertices).map(|v| v as u32).collect();
    vertices.sort_unstable_by_key(|&v| graph.degree(v));

    let mut colors: Vec<u32> = vec![std::u32::MAX; nvertices];
    let mut has_been_colored = vec![false; nvertices];
    let mut adjacent_colors: Vec<bool> = Vec::new();

    for vertex in vertices.into_iter().rev() {
        // loop invariant is that none of adjacent_colors elements are true

        // what color are our neighbors?
        let mut nadjacent_colors = 0;
        for &n in graph.neighbors(vertex) {
            let n = n as usize;
            if !has_been_colored[n] {
                continue;
            }

            let c = colors[n] as usize;
            if !adjacent_colors[c] {
                adjacent_colors[c] = true;
                nadjacent_colors += 1;
                if nadjacent_colors == colors.len() {
                    break;
                }
            }
        }

        // what's the smallest color not in our neighbors?
        let chosen = if nadjacent_colors == adjacent_colors.len() {
            adjacent_colors.push(false);
            adjacent_colors.len() - 1
        } else {
            adjacent_colors.iter().copied().position(|x| !x).unwrap()
        };
        colors[vertex as usize] = chosen as u32;
        has_been_colored[vertex as usize] = true;

        // retain loop invariant, unset neighbor colors
        if graph.degree(vertex) >= adjacent_colors.len() {
            graph
                .neighbors(vertex)
                .iter()
                .copied()
                .flat_map(|n| {
                    let n = n as usize;
                    if has_been_colored[n] {
                        Some(colors[n])
                    } else {
                        None
                    }
                })
                .for_each(|c| {
                    adjacent_colors[c as usize] = false;
                });
        } else {
            for c in adjacent_colors.iter_mut() {
                *c = false;
            }
        }
    }

    let ncolors = adjacent_colors.len();
    let mut color_counts = vec![0u32; ncolors];
    colors.iter().copied().for_each(|c| {
        color_counts[c as usize] += 1;
    });

    let code: Vec<_> = (0..ncolors)
        .sorted_by_key(|i| -(color_counts[*i] as i64))
        .collect();
    let mut recode = vec![0u32; ncolors];
    for i in 0..ncolors {
        recode[code[i]] = i as u32;
    }

    for color in colors.iter_mut() {
        *color = recode[*color as usize];
    }

    (ncolors as u32, colors)
}

fn write_encoding(
    encoding_dictionary: &EncodingDictionary,
    scanner: &SvmScanner,
    out_suffix: &str,
    featurizer: &Featurizer,
    is_train: bool,
) {
    scanner.for_each_sink(
        |mut word_iter, file, (dense, encoder): &mut (Vec<f64>, Encoder<'_>)| {
            if is_train && encoder.skip_example(word_iter.clone()) {
                return;
            }

            file.write_all(word_iter.next().expect("target"))
                .expect("successful write");

            featurizer.write_dense(word_iter.clone(), dense);

            for h in word_iter
                .enumerate()
                .flat_map(|(i, word)| featurizer.sparse(i, word))
            {
                encoder.observe(h)
            }

            for (i, value) in dense.iter().copied().enumerate() {
                if value == 0.0 {
                    continue;
                }
                write!(file, " {}:{}", i + encoder.dense_offset(), value)
                    .expect("successful write");
            }
            encoder.finish(file);
            file.write_all(b"\n").expect("successful write");
        },
        out_suffix,
        || {
            (
                vec![0.0f64; featurizer.ndense()],
                encoding_dictionary.new_encoder(),
            )
        },
    );
}

fn validate_coloring(ncolors: usize, colors: &[u32], train: &SvmScanner, featurizer: &Featurizer) {
    let (violation, _) = train.fold_reduce(
        || (None, vec![-1i64; ncolors]),
        |(error, mut cs), word_iter| {
            if error.is_some() {
                return (error, cs);
            }

            let original_line = word_iter.clone();
            let word_iter = word_iter.skip(1);
            let saved_words = word_iter.clone();
            for (i, h) in word_iter
                .enumerate()
                .flat_map(|(i, word)| featurizer.sparse(i, word).map(|w| (i, w)))
            {
                let color = colors[h as usize] as usize;
                if cs[color] >= 0 {
                    let s = original_line.dbg_line();
                    let mut saved_words = saved_words.skip(cs[color] as usize);
                    let word1 = saved_words.next().unwrap();
                    let j = i - (cs[color] as usize) - 1;
                    let mut saved_words = saved_words.skip(j);
                    let word2 = saved_words.next().unwrap();
                    // SAFETY: SvmScanner constructor makes caller promise
                    return (
                        Some((
                            "MULTIPLE COLORS {:?} ({}) and {:?} ({}) as {} in {}",
                            std::str::from_utf8(word1),
                            featurizer.sparse(j, word1).unwrap(),
                            std::str::from_utf8(word2),
                            featurizer.sparse(i, word2).unwrap(),
                            color,
                            s,
                        )),
                        cs,
                    );
                }
                cs[color] = i as i64;
            }
            for i in 0..ncolors {
                cs[i] = -1;
            }
            (None, cs)
        },
        |(a, _), (b, _)| (a.or(b), vec![]),
    );
    assert!(violation.is_none(), "{:?}", violation);
}

fn color_collision_count(
    valid: &SvmScanner,
    featurizer: &Featurizer,
    colors: &[u32],
) -> (f64, f64) {
    let (color_collision_squared, color_collision_count, total_examples) = valid.fold_reduce(
        || (0usize, 0usize, 0usize),
        |(color_collision_squared, color_collision_count, total_examples), word_iter| {
            // skip the target word

            let mut indices = Vec::with_capacity(1024);
            for (i, word) in word_iter.skip(1).enumerate() {
                let h = match featurizer.sparse(i, word) {
                    Some(h) => h,
                    None => continue,
                };
                indices.push(h);
            }

            indices.sort_unstable();
            indices.dedup();
            for i in indices.iter_mut() {
                *i = colors[*i as usize];
            }
            let mut row_colors = indices;
            row_colors.sort_unstable();

            let nnz = row_colors.len();
            row_colors.dedup();
            let unique_colors = row_colors.len();

            let collision_count = nnz - unique_colors;

            (
                color_collision_squared + collision_count.pow(2),
                color_collision_count + collision_count,
                total_examples + 1,
            )
        },
        |(a1, a2, a3), (b1, b2, b3)| (a1 + b1, a2 + b2, a3 + b3),
    );
    // not stable, blah blah blah
    let avg = color_collision_count as f64 / total_examples as f64;
    let std = color_collision_squared as f64 / total_examples as f64;
    let std = (std - avg.powi(2)).sqrt();
    (avg, std)
}

/// Sample a coloring uniformly (and in parallel), with respect to the given graph.
///
/// Accepts a mask to operate on the induced subgraph of the given graph. Vertices
/// that are masked out are given unique colors. Uses `unif_colors` for the coloring,
/// which should be at least the number of colors used for a greedy coloring.
fn glauber_color(graph: &AdjacencyList, unif_colors: u32, mask: &[bool]) -> (u32, Vec<u32>) {
    let (ncolors, mut colors) = greedy_color(&graph);
    assert!(ncolors < unif_colors);

    // https://www.math.cmu.edu/~af1p/Texfiles/colorbdd.pdf
    // https://www.math.cmu.edu/~af1p/Teaching/MCC17/Papers/colorJ.pdf
    // run glauber markov chain on a coloring
    // chain sampling can be parallel with some simple conflict detection

    let max_degree = graph.max_degree(mask);
    let avg_degree = graph.avg_degree(mask);
    let subgraph_sz = mask.iter().copied().filter(|x| *x).count();

    // by expectation version of birthday problem, don't want more than this
    // much parallelism
    let max_parallel = ((subgraph_sz as f64).sqrt() / avg_degree as f64) as usize;
    let nthreads = (rayon::current_num_threads() as usize).min(max_parallel.max(16));

    println!(
        "chose parallelism {} (recommendation {}) for {}-coloring",
        nthreads, max_parallel, unif_colors
    );

    use std::sync::atomic::{AtomicU32, Ordering};

    let samples = max_degree * subgraph_sz;

    let colors = colors
        .into_iter()
        .map(|x| AtomicU32::new(x))
        .collect::<Vec<_>>();

    let conflicts = (0..nthreads)
        .into_par_iter()
        .map(|i| {
            let samples = (samples / nthreads).max(1);
            let mut rng = StdRng::seed_from_u64((1234 + i) as u64); // no idea how to do this better in rust
            let mut adjacent_color: Vec<bool> = vec![false; unif_colors as usize];
            let mut neighbor_colors: Vec<u32> = Vec::with_capacity(max_degree as usize);
            let mut conflicts = 0;

            for _ in 0..samples {
                // loop invariant is that none of adjacent_color elements are true

                let reservation = (unif_colors + 1) as u32;

                loop {
                    let v: u32 = rng.gen_range(0, graph.nvertices() as u32);
                    if !mask[v as usize] {
                        continue;
                    }

                    // "claim" this vertex
                    let prev = colors[v as usize].swap(reservation, Ordering::Relaxed);
                    if prev == reservation {
                        conflicts += 1;
                        continue;
                    }

                    let mut nadjacent_colors = 0;
                    let mut fail = v;
                    for w in graph.masked_neighbors(v, mask) {
                        let c = colors[w as usize].swap(reservation, Ordering::Relaxed);
                        if c == reservation {
                            fail = w;
                            break;
                        }
                        if !adjacent_color[c as usize] {
                            nadjacent_colors += 1;
                            adjacent_color[c as usize] = true;
                        }
                        neighbor_colors.push(c);
                    }
                    if fail != v {
                        // unlock
                        colors[v as usize].store(prev, Ordering::Relaxed);
                        graph
                            .masked_neighbors(v, mask)
                            .take_while(|&w| w != fail)
                            .zip(neighbor_colors.iter().copied())
                            .for_each(|(w, c)| colors[w as usize].store(c, Ordering::Relaxed));
                        conflicts += 1;
                        for i in adjacent_color.iter_mut() {
                            *i = false;
                        }
                        neighbor_colors.clear();
                        continue;
                    }

                    let chosen = adjacent_color
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, x)| !x)
                        .map(|(i, _)| i)
                        .nth(rng.gen_range(0, (unif_colors as usize) - nadjacent_colors))
                        .expect("nth");

                    graph
                        .masked_neighbors(v, mask)
                        .zip(neighbor_colors.iter().copied())
                        .for_each(|(w, c)| colors[w as usize].store(c, Ordering::Relaxed));

                    colors[v as usize].store(chosen as u32, Ordering::Relaxed);

                    for i in adjacent_color.iter_mut() {
                        *i = false;
                    }
                    neighbor_colors.clear();
                    break;
                }
            }
            conflicts
        })
        .sum::<usize>();

    println!(
        "sampled {}-coloring with {} MCMC steps, {:.1}% overhead on {} threads",
        unif_colors,
        samples,
        100.0 * conflicts as f64 / samples as f64,
        nthreads
    );

    let mut colors: Vec<_> = colors.into_iter().map(|x| x.into_inner()).collect();

    let mut ctr = unif_colors;
    for (c, m) in colors.iter_mut().zip(mask.iter().copied()) {
        if !m {
            *c = ctr;
            ctr += 1;
        }
    }

    (ctr, colors)
}
