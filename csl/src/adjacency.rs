//! Adjacency list construction for graphs

use crate::edges::{Edge, Vertex};
use itertools::EitherOrBoth::{Both, Right};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::convert::TryInto;
use std::iter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// A compact adjacency list intended for sparse graphs.
///
/// The space of vertices is a contiguous range of u32 ints
/// from [0, nvertices).
pub(crate) struct AdjacencyList {
    offsets: Vec<usize>,
    all_neighbors: Vec<Vertex>,
    freqs: Vec<usize>, // counts of appearance for each edge
}

/// Note that masked methods behave as if they're being called on the corresponding induced
/// subgraph.
impl AdjacencyList {
    pub(crate) fn new(nvertices: usize, mut edges: Vec<Edge>, freqs: Vec<usize>) -> AdjacencyList {
        assert!(nvertices <= (1 << 32));

        let start = Instant::now();
        let offsets: Vec<_> = iter::repeat_with(|| AtomicUsize::new(0))
            .take(nvertices + 1)
            .collect();
        edges.par_iter().for_each(|e| {
            let (l, r) = (e.left(), e.right());
            offsets[l as usize].fetch_add(1, Ordering::Relaxed);
            offsets[r as usize].fetch_add(1, Ordering::Relaxed);
        });
        println!(
            "adjacency degree {:.0?}",
            Instant::now().duration_since(start)
        );

        let max_degree = offsets[..offsets.len() - 1]
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0);
        let avg_degree = offsets[..offsets.len() - 1]
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .sum::<usize>() as f64
            / nvertices as f64;
        println!("max degree {} avg degree {:.1}", max_degree, avg_degree);

        let start = Instant::now();
        let offsets: Vec<_> = offsets
            .into_iter()
            .scan(0usize, |state, next| {
                let value = *state;
                *state += next.into_inner();
                Some(value)
            })
            .collect();
        let last = offsets.len() - 1;
        assert!(offsets[0] == 0);
        assert!(offsets[last] == edges.len() * 2);
        println!(
            "adjacency offsets {:.0?}",
            Instant::now().duration_since(start)
        );

        let start = Instant::now();
        let nthreads = rayon::current_num_threads();
        // can probably do the trick in edges.rs by slicing up a vector
        let mut destinations = Vec::with_capacity(nthreads);
        let mut start_idx = 0;
        for i in 1..=nthreads {
            let end = i * (edges.len() * 2) / nthreads;
            for j in start_idx..offsets.len() {
                if offsets[j] >= end {
                    let start = offsets[start_idx];
                    let local_offsets: Vec<_> = offsets[start_idx..=j]
                        .iter()
                        .copied()
                        .map(|o| o - start)
                        .collect();
                    destinations.push((start_idx, j, local_offsets));
                    start_idx = j;
                    break;
                }
            }
        }
        assert!(destinations.len() == nthreads);
        assert!(destinations[0].0 == 0);
        assert!(offsets[destinations[nthreads - 1].1] == edges.len() * 2);
        println!("destinations {:.0?}", Instant::now().duration_since(start));

        let (bidir_edges, bidir_freqs): (Vec<_>, Vec<_>) = destinations
            .into_par_iter()
            .map(|(lo, hi, mut local_offsets)| {
                let last = local_offsets.len() - 1;
                assert!(last == hi - lo);
                let mut bidir_edges = vec![std::u32::MAX; local_offsets[last]];
                let mut bidir_freqs = vec![std::usize::MAX; local_offsets[last]];
                for (e, f) in edges.iter().zip(freqs.iter()) {
                    let (l, r) = (e.left() as usize, e.right() as usize);
                    if lo <= l && l < hi {
                        bidir_edges[local_offsets[l - lo]] = r as Vertex;
                        bidir_freqs[local_offsets[l - lo]] = *f;
                        local_offsets[l - lo] += 1;
                    }
                    if lo <= r && r < hi {
                        bidir_edges[local_offsets[r - lo]] = l as Vertex;
                        bidir_freqs[local_offsets[r - lo]] = *f;
                        local_offsets[r - lo] += 1;
                    }
                }
                (bidir_edges, bidir_freqs)
            })
            .unzip();
        let mut bidir_edges: Vec<_> = bidir_edges.into_iter().flatten().collect();
        let bidir_freqs: Vec<_> = bidir_freqs.into_iter().flatten().collect();

        if cfg!(debug_assertions) {
            validate_edges(&mut edges, &mut bidir_edges, &offsets);
        }

        println!(
            "adjacency assign {:.0?}",
            Instant::now().duration_since(start)
        );
        AdjacencyList {
            offsets,
            all_neighbors: bidir_edges,
            freqs: bidir_freqs,
        }
    }

    pub(crate) fn neighbors(&self, v: Vertex) -> &[Vertex] {
        let v = v as usize;
        let lo = self.offsets[v];
        let hi = self.offsets[v + 1];
        &self.all_neighbors[lo..hi]
    }

    pub(crate) fn masked_neighbors<'a>(
        &'a self,
        v: Vertex,
        mask: &'a [bool],
    ) -> impl Iterator<Item = Vertex> + 'a {
        let v = v as usize;
        let lo = self.offsets[v];
        let hi = self.offsets[v + 1];
        self.all_neighbors[lo..hi]
            .iter()
            .copied()
            .filter(move |&v| mask[v as usize])
    }

    pub(crate) fn degree(&self, v: Vertex) -> usize {
        let v = v as usize;
        let lo = self.offsets[v];
        let hi = self.offsets[v + 1];
        hi - lo
    }

    pub(crate) fn nvertices(&self) -> usize {
        self.offsets.len() - 1
    }

    /// How many edges appeared exactly k times?
    pub(crate) fn appeared_k(&self, k: usize, mask: &[bool]) -> usize {
        self.iter_edges()
            .filter(|(e, f)| *f == k && mask[e.left() as usize] && mask[e.right() as usize])
            .count()
    }

    pub(crate) fn internal_sort(&mut self) {
        for v in 0..self.nvertices() {
            let lo = self.offsets[v];
            let hi = self.offsets[v + 1];
            // really dumb permutation but w/e it's for debug
            let mut ixs: Vec<_> = (lo..hi).collect();
            ixs.sort_unstable_by_key(|ix| self.all_neighbors[*ix]);
            let cpy: Vec<_> = self.all_neighbors[lo..hi].iter().copied().collect();
            for (i, ix) in (lo..hi).zip(ixs.iter().copied()) {
                self.all_neighbors[i] = cpy[ix - lo];
            }
            let cpy: Vec<_> = self.freqs[lo..hi].iter().copied().collect();
            for (i, ix) in (lo..hi).zip(ixs.iter().copied()) {
                self.freqs[i] = cpy[ix - lo];
            }
        }
    }

    fn iter_edges<'a>(&'a self) -> impl Iterator<Item = (Edge, usize)> + 'a {
        self.offsets
            .iter()
            .copied()
            .zip(self.offsets.iter().skip(1).copied())
            .enumerate()
            .flat_map(move |(v, (lo, hi))| {
                let v = v as Vertex;
                (lo..hi)
                    .filter(move |&i| self.all_neighbors[i] > v)
                    .map(move |i| (Edge::new(v, self.all_neighbors[i]), self.freqs[i]))
            })
    }

    /// Given a list of edges and the number of times they were observed `other`,
    /// this returns how many times an edge in `other` appeared less than k times (including 0)
    /// in this graph.
    ///
    /// Graphs must be sorted and over the same vertex set.
    pub(crate) fn nmissing(&self, k: usize, other: &AdjacencyList, mask: &[bool]) -> usize {
        assert!(
            self.nvertices() == other.nvertices(),
            "vertex counts {} != {}",
            self.nvertices(),
            other.nvertices()
        );
        self.iter_edges()
            .merge_join_by(other.iter_edges(), |(l, _), (r, _)| l.cmp(r))
            // we only care about situations where the edge is present in
            // other but is missing in self or only present in self less than k times
            .flat_map(|either| match either {
                Right((e, cnt)) => Some((e, cnt)),
                Both((e, i), (e2, cnt)) if i < k => {
                    assert!(e == e2);
                    Some((e, cnt))
                }
                _ => None,
            })
            // if an edge contains a masked-out endpoint then ignore it
            .map(|(e, cnt)| {
                let l = e.left() as usize;
                let r = e.right() as usize;
                if mask[l] && mask[r] {
                    cnt
                } else {
                    0
                }
            })
            .sum::<usize>()
    }

    /// Filter a graph to edges that occur at least k times. Note this may result in
    /// isolated vertices.
    pub(crate) fn filter(&self, k: usize) -> AdjacencyList {
        let mut new_offsets = self.offsets.clone();
        let mut accum = 0;
        for v in 0..self.nvertices() {
            for i in self.offsets[v]..self.offsets[v + 1] {
                if self.freqs[i] < k {
                    accum += 1;
                }
            }
            new_offsets[v + 1] -= accum;
        }
        let (nbrs, freqs): (Vec<_>, Vec<_>) = self
            .all_neighbors
            .iter()
            .copied()
            .zip(self.freqs.iter().copied())
            .filter(|(_, f)| *f >= k)
            .unzip();
        Self {
            offsets: new_offsets,
            all_neighbors: nbrs,
            freqs,
        }
    }

    pub(crate) fn max_degree(&self, mask: &[bool]) -> usize {
        (0..self.nvertices())
            .filter(|&v| mask[v])
            .map(|v| self.degree(v as Vertex))
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn avg_degree(&self, mask: &[bool]) -> f64 {
        let (nverts, sum) = (0..self.nvertices())
            .filter(|&v| mask[v])
            .map(|v| (1, self.degree(v as Vertex)))
            .fold((0, 0), |(a, b), (c, d)| (a + c, b + d));
        sum as f64 / nverts as f64
    }

    /// Return the maximum and average degree of the graph that would be returned by filter(k)
    pub(crate) fn filter_degree(&self, k: usize) -> (usize, f64) {
        let mut max_degree = 0;
        let mut sum_degree = 0;
        for v in 0..self.nvertices() {
            let mut degree = 0;
            for i in self.offsets[v]..self.offsets[v + 1] {
                if self.freqs[i] >= k {
                    degree += 1;
                }
            }
            max_degree = max_degree.max(degree);
            sum_degree += degree;
        }
        (max_degree, sum_degree as f64 / self.nvertices() as f64)
    }

    /// Return the largest-first ordering lf, where lf[i].0 is the vertex of maximum
    /// degree in the induced subgraph over all vertices lf[i].0 to lf[lf.len() - 1].0.
    ///
    /// In the returned ordering, the first element of the tuple is the vertex, the second
    /// is its degree in the induced subgraph mentioned above.
    pub(crate) fn largest_first(&self) -> Vec<(Vertex, u32)> {
        // https://dl.acm.org/doi/pdf/10.1145/2402.322385
        // Smallest-Last Ordering and Clustering and Graph Coloring Algorithms
        // Matula and Beck 1983

        let mut return_order = Vec::with_capacity(self.nvertices());
        let nv = self.nvertices();
        let null = std::u32::MAX;

        if nv == 0 {
            return vec![];
        }

        // The Matula and Beck algorithm is inverted here.
        //
        // The algorithm proceeds with nv stages, one for each vertex.
        // At each i-th stage of the algorithm, we maintain:
        //
        // the implicit graph H over all unexplored vertices, defined by
        //   the induced graph from the negation of `explored` as a mask over the
        //   initial graph.
        //
        // `degrees[v]` the current degree of every unexplored vertex v in H
        //
        // `prio` is a priority array, which contains the list of unexplored vertices
        // sorted in ascending order by their degree within H
        //
        // `offsets` is defined such that `prio[offsets[i]..offsets[i+1]]` is exactly
        // all vertices with degree i in H. This holds for all i from 0 to curr_max,
        // where curr_max is the maximum degree in H.
        //
        // `iprio[v]` is the inverse of the priority array holding the index at which
        // vertex `v` is located in `prio` if `v` is unexplored.

        let mut explored = vec![false; nv];
        let mut max_degree = (0..nv).map(|v| self.degree(v as Vertex)).max().unwrap_or(0) as u32;
        let mut offsets = vec![0u32; u(max_degree) + 2];
        (0..nv).for_each(|v| offsets[self.degree(v as Vertex) + 1] += 1);
        cumsum(&mut offsets);
        let mut prio = vec![null; nv];
        let mut iprio = vec![null; nv];
        let mut degrees = vec![null; nv];
        {
            let mut fill_offsets = offsets.clone();
            for v in 0..nv {
                let deg = self.degree(v as Vertex);
                prio[u(fill_offsets[deg])] = v as Vertex;
                iprio[v] = fill_offsets[deg];
                degrees[v] = d(deg);
                fill_offsets[deg] += 1;
            }
        }

        while return_order.len() < nv {
            // pop off the max vertex

            assert!(offsets[u(max_degree)] < offsets[u(max_degree) + 1]);
            let last = offsets[u(max_degree) + 1] - 1;
            let max_vertex = prio[u(last)];

            assert!(iprio[u(max_vertex)] == last);
            assert!(!explored[u(max_vertex)]);
            assert!(degrees[u(max_vertex)] == max_degree);

            iprio[u(max_vertex)] = null;
            degrees[u(max_vertex)] = null;
            prio[u(last)] = null;

            // explore the max vertex

            explored[u(max_vertex)] = true;
            return_order.push((max_vertex, max_degree));

            // reduce unexplored neighbors' degrees by 1

            let mut nbrs = 0;
            for &v in self.neighbors(max_vertex) {
                if explored[u(v)] {
                    continue;
                }

                nbrs += 1;
                assert!(max_degree != null); // no neighbors if null

                let deg = degrees[u(v)];
                let ix = iprio[u(v)];
                assert!(deg > 0); // connected to max_vertex before

                let target_ix = offsets[u(deg)];
                let target = prio[u(target_ix)];
                assert!(iprio[u(target)] == target_ix);
                assert!(!explored[u(target)]);

                // swap with target

                prio[u(target_ix)] = v;
                iprio[u(v)] = target_ix;
                degrees[u(v)] = deg - 1;

                prio[u(ix)] = target;
                iprio[u(target)] = ix;
                assert!(degrees[u(target)] == deg || target == v);

                // move the offset to match new degrees

                offsets[u(deg)] += 1;
            }
            assert!(nbrs == max_degree);

            // reduce the new current maximum degree

            offsets[u(max_degree) + 1] -= 1;
            while offsets[u(max_degree)] == offsets[u(max_degree) + 1] {
                offsets[u(max_degree) + 1] = null;
                if max_degree > 0 {
                    max_degree -= 1;
                } else {
                    max_degree = null;
                    break;
                }
            }
        }

        assert!(max_degree == null);

        return_order
    }
}

fn u(x: u32) -> usize {
    x as usize
}

fn d(x: usize) -> u32 {
    x.try_into().unwrap()
}

fn cumsum(v: &mut [u32]) {
    let mut running_sum = 0u32;
    for i in v.iter_mut() {
        running_sum += *i;
        *i = running_sum;
    }
}

fn validate_edges(edges: &mut Vec<Edge>, bidir_edges: &mut Vec<Vertex>, offsets: &[usize]) {
    let nvertices = offsets.len() - 1;

    edges.par_sort_unstable();
    let mut adj_edges = Vec::with_capacity(edges.len());
    for left in 0..nvertices {
        bidir_edges[offsets[left]..offsets[left + 1]].sort_unstable();
        for right_ix in offsets[left]..offsets[left + 1] {
            let right: u32 = bidir_edges[right_ix];
            let left = left as u32;
            if left < right {
                adj_edges.push(Edge::new(left, right));
            }
        }
    }
    for (left, right) in adj_edges.iter().zip(edges.iter()) {
        assert!(
            left == right,
            "({},{}) != ({},{})",
            left.left(),
            left.right(),
            right.left(),
            right.right()
        );
    }
}
