//! Submodular optimization module.

use rand::Rng;

/// This trait describes a partial solution to a submodular optimization problem.
///
/// Submodular functions f are set-valued over some finite universe A.
/// In this context, we'll assume they are monotonic. This means
///
/// f(A) <= f(B) if A <= B (<= denoting set inclusion) [montonicity]
///
/// f(A + a) - f(A) >= f(B + a) - f(B) if A <= B (+ being union) [submodularity]
///
/// Such functions admit efficient approximate optimization routines. [1, 2]
///
/// This trait defines the key operations expected from an implementation
/// of partial solution sets A.
///
///  1. gain(x) == f(A + x) - f(A)
///  2. insert(x) adds x to current set A
///
/// As such, the trait defines both the function f (implicitly) and a "working solution".
///
/// [1] https://homes.cs.washington.edu/~marcotcr/blog/greedy-submodular/
/// [2] https://homes.cs.washington.edu/~marcotcr/blog/lazier/
pub trait Submodular<A> {
    fn gain(a: &A) -> f64;

    fn insert(a: A);
}

/// Increases the given submodular function solution by k items (at most,
/// will not add items if they don't improve gains). Picks items from
/// the given universe to add, where the universe is given by an iterator
/// over its elements (only ever run once)
///
/// Reports total gains since the start, using the Lazy Greedy algorithm.
#[allow(dead_code)]
pub fn lazy_greedy<A, S>(_s: &mut S, _universe: impl Iterator<Item = A>) -> f64
where
    S: Submodular<A>,
{
    0.
}

/// Same as lazy greedy, but uses the Stochastic Greedy algorithm.
#[allow(dead_code)]
pub fn stochastic_greedy<A, S, R>(_s: &mut S, _universe: impl Iterator<Item = A>, _r: &mut R) -> f64
where
    S: Submodular<A>,
    R: Rng + ?Sized,
{
    0.
}
