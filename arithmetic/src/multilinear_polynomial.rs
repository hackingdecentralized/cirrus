// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use crate::{start_timer_with_timestamp, util::get_batched_nv, ArithErrors};
use ark_ff::{Field, PrimeField};
use ark_poly::MultilinearExtension;
use ark_std::{end_timer, rand::RngCore};
#[cfg(feature = "parallel")]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{cmp::min, sync::Arc};

pub use ark_poly::DenseMultilinearExtension;

/// Sample a random list of multilinear polynomials.
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube.
pub fn random_mle_list<F: PrimeField, R: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> (Vec<Arc<DenseMultilinearExtension<F>>>, F) {
    let start = start_timer_with_timestamp!("sample random mle list");
    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    let mut sum = F::zero();

    for _ in 0..(1 << nv) {
        let mut product = F::one();

        for e in multiplicands.iter_mut() {
            let val = F::rand(rng);
            e.push(val);
            product *= val;
        }
        sum += product;
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();

    end_timer!(start);
    (list, sum)
}

/// Split a multilinear polynomial into a list of polynomials.
/// Input: a multilinear polynomial `poly`
///        a log of the number of chunks `log_num_chunks`
/// Requires: variable of poly is larger than log_num_chunks
pub fn split_into_chunks<F: PrimeField>(
    poly: &Arc<DenseMultilinearExtension<F>>,
    log_num_chunks: usize,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let nv = poly.num_vars();
    let log_num_chunks = min(log_num_chunks, nv);
    let num_chunks = 1 << log_num_chunks;
    let mut res = vec![];
    let n = 1 << (nv - log_num_chunks);
    for i in 0..num_chunks {
        res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            nv - log_num_chunks,
            poly.evaluations[i * n..i * n + n].to_vec(),
        )));
    }
    res
}

// Build a randomize list of mle-s whose sum is zero.
pub fn random_zero_mle_list<F: PrimeField, R: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut R,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let start = start_timer_with_timestamp!("sample random zero mle list");

    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    for _ in 0..(1 << nv) {
        multiplicands[0].push(F::zero());
        for e in multiplicands.iter_mut().skip(1) {
            e.push(F::rand(rng));
        }
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();

    end_timer!(start);
    list
}

pub fn identity_permutation<F: PrimeField>(num_vars: usize, num_chunks: usize) -> Vec<F> {
    let len = (num_chunks as u64) * (1u64 << num_vars);
    (0..len).map(F::from).collect()
}

/// A list of MLEs that represents an identity permutation
pub fn identity_permutation_mles<F: PrimeField>(
    num_vars: usize,
    num_chunks: usize,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let mut res = vec![];
    for i in 0..num_chunks {
        let shift = (i * (1 << num_vars)) as u64;
        let s_id_vec = (shift..shift + (1u64 << num_vars)).map(F::from).collect();
        res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, s_id_vec,
        )));
    }
    res
}

pub fn random_permutation<R: RngCore>(num_vars: usize, num_chunks: usize, rng: &mut R) -> Vec<u64> {
    let len = (num_chunks as u64) * (1u64 << num_vars);
    let mut s_id_vec: Vec<u64> = (0..len).collect();
    let mut s_perm_vec = vec![];
    for _ in 0..len {
        let index = rng.next_u64() as usize % s_id_vec.len();
        s_perm_vec.push(s_id_vec.remove(index));
    }
    s_perm_vec
}

/// A list of MLEs that represent a random permutation
pub fn random_permutation_mles<F: PrimeField, R: RngCore>(
    num_vars: usize,
    num_chunks: usize,
    rng: &mut R,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let s_perm_vec: Vec<_> = random_permutation(num_vars, num_chunks, rng)
        .iter()
        .cloned()
        .map(F::from)
        .collect();
    let mut res = vec![];
    let n = 1 << num_vars;
    for i in 0..num_chunks {
        res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            s_perm_vec[i * n..i * n + n].to_vec(),
        )));
    }
    res
}

pub fn random_permutation_with_corresponding_mles<F: PrimeField, R: RngCore>(
    num_vars: usize,
    num_chunks: usize,
    rng: &mut R,
) -> (
    Vec<Arc<DenseMultilinearExtension<F>>>,
    Vec<Arc<DenseMultilinearExtension<F>>>,
    Vec<Arc<DenseMultilinearExtension<F>>>,
) {
    let total_num = num_chunks * (1 << num_vars);
    let perm = random_permutation(num_vars, num_chunks, rng);
    let mut set = vec![false; total_num];
    let mut evals = vec![F::zero(); total_num];

    for x in 0..total_num {
        if set[x] {
            continue;
        }

        let val = F::rand(rng);
        evals[x] = val;
        let mut cur = perm[x] as usize;
        while cur != x {
            evals[cur] = val;
            set[cur] = true;
            cur = perm[cur] as usize;
        }
    }

    let res = (
        (0..total_num)
            .map(|x| F::from(x as u128))
            .collect::<Vec<_>>(),
        perm.iter().cloned().map(F::from).collect::<Vec<_>>(),
        evals,
    );

    let f = |x: &Vec<F>| {
        x.chunks(1 << num_vars)
            .map(|x| {
                Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                    num_vars, x,
                ))
            })
            .collect()
    };

    (f(&res.0), f(&res.1), f(&res.2))
}

pub fn evaluate_opt<F: Field>(poly: &DenseMultilinearExtension<F>, point: &[F]) -> F {
    assert_eq!(poly.num_vars, point.len());
    fix_variables(poly, point).evaluations[0]
}

pub fn fix_variables<F: Field>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for (i, point) in partial_point.iter().enumerate().take(dim) {
        poly = fix_one_variable_helper(&poly, nv - i, point);
    }

    DenseMultilinearExtension::<F>::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

fn fix_one_variable_helper<F: Field>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let mut res = vec![F::zero(); 1 << (nv - 1)];

    // evaluate single variable of partial point from left to right
    #[cfg(not(feature = "parallel"))]
    for i in 0..(1 << (nv - 1)) {
        res[i] = data[i] + (data[(i << 1) + 1] - data[i << 1]) * point;
    }

    #[cfg(feature = "parallel")]
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i << 1] + (data[(i << 1) + 1] - data[i << 1]) * point;
    });

    res
}

pub fn evaluate_no_par<F: Field>(poly: &DenseMultilinearExtension<F>, point: &[F]) -> F {
    assert_eq!(poly.num_vars, point.len());
    fix_variables_no_par(poly, point).evaluations[0]
}

fn fix_variables_no_par<F: Field>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for i in 1..dim + 1 {
        let r = partial_point[i - 1];
        for b in 0..(1 << (nv - i)) {
            poly[b] = poly[b << 1] + (poly[(b << 1) + 1] - poly[b << 1]) * r;
        }
    }
    DenseMultilinearExtension::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

/// merge a set of polynomials. Returns an error if the
/// polynomials do not share a same number of nvs.
pub fn merge_polynomials<F: PrimeField>(
    polynomials: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<Arc<DenseMultilinearExtension<F>>, ArithErrors> {
    let nv = polynomials[0].num_vars();
    for poly in polynomials.iter() {
        if nv != poly.num_vars() {
            return Err(ArithErrors::InvalidParameters(
                "num_vars do not match for polynomials".to_string(),
            ));
        }
    }

    let merged_nv = get_batched_nv(nv, polynomials.len());
    let mut scalars = vec![];
    for poly in polynomials.iter() {
        scalars.extend_from_slice(poly.to_evaluations().as_slice());
    }
    scalars.extend_from_slice(vec![F::zero(); (1 << merged_nv) - scalars.len()].as_ref());
    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        merged_nv, scalars,
    )))
}

pub fn fix_last_variables_no_par<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    let mut res = fix_last_variable_no_par(poly, partial_point.last().unwrap());
    for p in partial_point.iter().rev().skip(1) {
        res = fix_last_variable_no_par(&res, p);
    }
    res
}

fn fix_last_variable_no_par<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &F,
) -> DenseMultilinearExtension<F> {
    let nv = poly.num_vars();
    let half_len = 1 << (nv - 1);
    let mut res = vec![F::zero(); half_len];
    for (i, e) in res.iter_mut().enumerate().take(half_len) {
        *e = poly.evaluations[i]
            + *partial_point * (poly.evaluations[i + half_len] - poly.evaluations[i]);
    }
    DenseMultilinearExtension::from_evaluations_vec(nv - 1, res)
}
pub fn fix_last_variables<F: PrimeField>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for (i, point) in partial_point.iter().rev().enumerate().take(dim) {
        poly = fix_last_variable_helper(&poly, nv - i, point);
    }

    DenseMultilinearExtension::<F>::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}

fn fix_last_variable_helper<F: Field>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let half_len = 1 << (nv - 1);
    let mut res = vec![F::zero(); half_len];

    // evaluate single variable of partial point from left to right
    #[cfg(not(feature = "parallel"))]
    for b in 0..half_len {
        res[b] = data[b] + (data[b + half_len] - data[b]) * point;
    }

    #[cfg(feature = "parallel")]
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i] + (data[i + half_len] - data[i]) * point;
    });

    res
}
