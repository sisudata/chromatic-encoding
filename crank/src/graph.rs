//! Compact graph data structure.

use std::convert::TryInto;

use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;

pub(crate) type Vertex = u32;

/// A compact adjacency list intended for sparse graphs.
///
/// The space of vertices is a contiguous range of u32 ints
/// from [0, nvertices).
pub struct Graph {
    offsets: Vec<usize>,
    neighbors: Vec<Vertex>,
}

impl Graph {
    /// `offsets.len()` should be one greater than the number of vertices
    /// with `neighbors[offsets[i]..offsets[i+1]]` being the edges incident
    /// from `i`, which should be necessarily sorted and bidirectional.
    pub(crate) fn new(offsets: Vec<usize>, neighbors: Vec<Vertex>) -> Self {
        assert!(offsets.len() <= (1 << 32));
        debug_assert!(offsets.par_windows(2).enumerate().all(|(i, s)| {
            s[0] < s[1]
                && neighbors[s[0]..s[1]].windows(2).all(|ss| ss[0] < ss[1])
                && neighbors[s[0]..s[1]].iter().copied().all(|j| {
                    let i = &(i as u32);
                    neighbors[offsets[j as usize]..offsets[1 + j as usize]]
                        .binary_search(i)
                        .is_ok()
                })
        }));
        Self { offsets, neighbors }
    }

    pub(crate) fn neighbors(&self, v: Vertex) -> &[Vertex] {
        let v = v as usize;
        let lo = self.offsets[v];
        let hi = self.offsets[v + 1];
        &self.neighbors[lo..hi]
    }

    pub(crate) fn degree(&self, v: Vertex) -> usize {
        let v = v as usize;
        let lo = self.offsets[v];
        let hi = self.offsets[v + 1];
        hi - lo
    }

    pub fn nvertices(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn subgraph(&self, ixs: &[u32]) -> Self {
        assert!(ixs.windows(2).all(|s| s[0] < s[1]));
        assert!(ixs
            .last()
            .iter()
            .all(|ix| (**ix as usize) < self.nvertices()));

        let mut remap = vec![u32::MAX; self.nvertices()];
        for (i, ix) in ixs.iter().copied().enumerate() {
            remap[ix as usize] = i.try_into().unwrap();
        }
        let mut new_offsets = vec![0usize; ixs.len() + 1];
        for &ix in ixs {
            let down_ix = remap[ix as usize] as usize;
            new_offsets[down_ix + 1] += self
                .neighbors(ix)
                .iter()
                .filter(|nbr| remap[**nbr as usize] != u32::MAX)
                .count();
        }
        cumsum_inplace(&mut new_offsets);

        let mut new_neighbors = vec![u32::MAX; *new_offsets.last().expect("last")];
        for &ix in ixs {
            let down_ix = remap[ix as usize] as usize;
            let (lo, hi) = (new_offsets[down_ix], new_offsets[down_ix + 1]);
            let new_nbr_cnt = self
                .neighbors(ix)
                .iter()
                .map(|ix| remap[*ix as usize])
                .filter(|ix| *ix != u32::MAX)
                .count();
            assert!(hi - lo == new_nbr_cnt);
            for (new_nbr, old_nbr) in new_neighbors[lo..hi].iter_mut().zip(
                self.neighbors(ix)
                    .iter()
                    .map(|ix| remap[*ix as usize])
                    .filter(|ix| *ix != u32::MAX),
            ) {
                *new_nbr = old_nbr;
            }
        }

        let ret = Self {
            offsets: new_offsets,
            neighbors: new_neighbors,
        };

        // println!(
        //     "{}",
        //     json!({
        //         "orig_degrees": SummaryStats::from(ixs.iter().map(|ix| self.degree(*ix) as f64)).to_map(),
        //         "subgraph_degrees": SummaryStats::from((0..ixs.len()).map(|ix| ret.degree(ix as u32) as f64)).to_map(),
        //     }
        //     )
        // );

        ret
    }
}

fn cumsum_inplace(x: &mut [usize]) {
    let mut cumsum = 0;
    for v in x.iter_mut() {
        cumsum += *v;
        *v = cumsum;
    }
}
