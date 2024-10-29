// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use crate::{
    custom_gate::CustomizedGates, errors::HyperPlonkErrors, structs::HyperPlonkParams,
    witness::WitnessColumn,
};
use arithmetic::{evaluate_opt, transpose, VPAuxInfo, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use std::{borrow::Borrow, marker::PhantomData, sync::Arc};
use subroutines::{
    pcs::{prelude::Commitment, PolynomialCommitmentScheme, PolynomialCommitmentSchemeDistributed},
    MasterProverChannel, WorkerProverChannel,
};
use transcript::IOPTranscript;

/// An accumulator structure that holds a polynomial and
/// its opening points
#[derive(Debug)]
pub(super) struct PcsAccumulator<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    // sequence:
    // - prod(x) at 5 points
    // - w_merged at perm check point
    // - w_merged at zero check points (#witness points)
    // - selector_merged at zero check points (#selector points)
    // - w[0] at r_pi
    pub(crate) num_var: usize,
    pub(crate) polynomials: Vec<PCS::Polynomial>,
    pub(crate) commitments: Vec<PCS::Commitment>,
    pub(crate) points: Vec<PCS::Point>,
    pub(crate) evals: Vec<PCS::Evaluation>,
}

impl<E, PCS> PcsAccumulator<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
    >,
{
    /// Create an empty accumulator.
    pub(super) fn new(num_var: usize) -> Self {
        Self {
            num_var,
            polynomials: vec![],
            commitments: vec![],
            points: vec![],
            evals: vec![],
        }
    }

    /// Push a new evaluation point into the accumulator
    pub(super) fn insert_poly_and_points(
        &mut self,
        poly: &PCS::Polynomial,
        commit: &PCS::Commitment,
        point: &PCS::Point,
    ) {
        assert!(poly.num_vars == point.len());
        assert!(poly.num_vars == self.num_var);

        let eval = evaluate_opt(poly, point);

        self.evals.push(eval);
        self.polynomials.push(poly.clone());
        self.points.push(point.clone());
        self.commitments.push(*commit);
    }

    /// Batch open all the points over a merged polynomial.
    /// A simple wrapper of PCS::multi_open
    pub(super) fn multi_open(
        &self,
        prover_param: impl Borrow<PCS::ProverParam>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<PCS::BatchProof, HyperPlonkErrors> {
        Ok(PCS::multi_open(
            prover_param.borrow(),
            self.polynomials.as_ref(),
            self.points.as_ref(),
            self.evals.as_ref(),
            transcript,
        )?)
    }
}

#[derive(Debug)]
pub struct PcsAccumulatorMaster<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<E>,
{
    pub num_var: usize,
    pub log_num_workers: usize,
    pub points: Vec<PCS::Point>,
    pub evals: Vec<E::ScalarField>,
}

impl<E, PCS> PcsAccumulatorMaster<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
        MasterPolynomialHandle = usize,
        WorkerPolynomialHandle = Arc<DenseMultilinearExtension<E::ScalarField>>,
    >,
{
    pub fn new(num_var: usize, log_num_workers: usize) -> Self {
        Self {
            num_var,
            log_num_workers,
            points: vec![],
            evals: vec![],
        }
    }

    pub fn insert_point(&mut self, point: PCS::Point) {
        assert!(point.len() == self.num_var);
        self.points.push(point);
    }

    pub fn eval_poly_and_points(
        &mut self,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<(), HyperPlonkErrors> {
        let worker_num_vars = self.num_var - self.log_num_workers;
        let mut worker_points = Vec::new();
        for point in self.points.iter() {
            let (worker_point, _) = point.split_at(worker_num_vars);
            worker_points.push(worker_point.to_vec());
        }

        master_channel.send_uniform(&worker_points)?;
        let evals: Vec<Vec<PCS::Evaluation>> = master_channel.recv()?;
        let evals = transpose(evals)
            .into_iter()
            .zip(self.points.iter())
            .map(|(evals, point)| {
                let (_, master_point) = point.split_at(worker_num_vars);
                evaluate_opt(
                    &DenseMultilinearExtension::from_evaluations_vec(self.log_num_workers, evals),
                    master_point,
                )
            })
            .collect::<Vec<_>>();

        self.evals = evals;
        Ok(())
    }

    pub fn multi_open(
        &self,
        prover_param: impl Borrow<PCS::MasterProverParam>,
        transcript: &mut IOPTranscript<E::ScalarField>,
        master_channel: &impl MasterProverChannel,
    ) -> Result<PCS::BatchProof, HyperPlonkErrors> {
        unimplemented!()
    }
}

pub struct PcsAccumulatorWorker<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<E>,
{
    pub num_var: usize,
    pub polynomials: Vec<PCS::WorkerPolynomialHandle>,
    // pub evals: Vec<E::ScalarField>,
}

impl<E, PCS> PcsAccumulatorWorker<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
        MasterPolynomialHandle = usize,
        WorkerPolynomialHandle = Arc<DenseMultilinearExtension<E::ScalarField>>,
    >,
{
    pub fn new(num_var: usize) -> Self {
        Self {
            num_var,
            polynomials: vec![],
            // evals: vec![],
        }
    }

    pub fn insert_poly(&mut self, poly: &PCS::WorkerPolynomialHandle) {
        assert!(poly.num_vars == self.num_var);
        self.polynomials.push(poly.clone());
    }

    pub fn eval_poly_and_points(
        &mut self,
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(), HyperPlonkErrors> {
        let points: Vec<PCS::Point> = worker_channel.recv()?;
        let evals = points
            .iter()
            .zip(self.polynomials.iter())
            .map(|(point, poly)| evaluate_opt(poly, point))
            .collect::<Vec<_>>();
        worker_channel.send(&evals)?;
        Ok(())
    }

    pub fn multi_open(
        &self,
        prover_param: impl Borrow<PCS::WorkerProverParam>,
        worker_channel: &impl WorkerProverChannel,
    ) -> Result<(), HyperPlonkErrors> {
        unimplemented!()
    }
}

/// Build MLE from matrix of witnesses.
///
/// Given a matrix := [row1, row2, ...] where
/// row1:= (a1, a2, ...)
/// row2:= (b1, b2, ...)
/// row3:= (c1, c2, ...)
///
/// output mle(a1,b1,c1, ...), mle(a2,b2,c2, ...), ...
#[macro_export]
macro_rules! build_mle {
    ($rows:expr) => {{
        let mut res = Vec::with_capacity($rows.len());
        let num_vars = log2($rows.len()) as usize;
        let num_mles = $rows[0].0.len();

        for i in 0..num_mles {
            let mut cur_coeffs = Vec::new();
            for row in $rows.iter() {
                cur_coeffs.push(row.0[i])
            }
            res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, cur_coeffs,
            )))
        }

        Ok(res)
    }};
}

/// Sanity-check for HyperPlonk SNARK proving
pub(crate) fn prover_sanity_check<F: PrimeField>(
    params: &HyperPlonkParams,
    pub_input: &[F],
    witnesses: &[WitnessColumn<F>],
) -> Result<(), HyperPlonkErrors> {
    // public input length must be no greater than num_constraints

    if pub_input.len() > params.num_constraints {
        return Err(HyperPlonkErrors::InvalidProver(format!(
            "Public input length {} is greater than num constraits {}",
            pub_input.len(),
            params.num_pub_input
        )));
    }

    // public input length
    if pub_input.len() != params.num_pub_input {
        return Err(HyperPlonkErrors::InvalidProver(format!(
            "Public input length is not correct: got {}, expect {}",
            pub_input.len(),
            params.num_pub_input
        )));
    }
    if !pub_input.len().is_power_of_two() {
        return Err(HyperPlonkErrors::InvalidProver(format!(
            "Public input length is not power of two: got {}",
            pub_input.len(),
        )));
    }

    // witnesses length
    for (i, w) in witnesses.iter().enumerate() {
        if w.0.len() != params.num_constraints {
            return Err(HyperPlonkErrors::InvalidProver(format!(
                "{}-th witness length is not correct: got {}, expect {}",
                i,
                w.0.len(),
                params.num_constraints
            )));
        }
    }
    // check public input matches witness[0]'s first 2^ell elements
    for (i, (&pi, &w)) in pub_input
        .iter()
        .zip(witnesses[0].0.iter().take(pub_input.len()))
        .enumerate()
    {
        if pi != w {
            return Err(HyperPlonkErrors::InvalidProver(format!(
                "The {:?}-th public input {:?} does not match witness[0] {:?}",
                i, pi, w
            )));
        }
    }

    Ok(())
}

/// build `f(w_0(x),...w_d(x))` where `f` is the constraint polynomial
/// i.e., `f(a, b, c) = q_l a(x) + q_r b(x) + q_m a(x)b(x) - q_o c(x)` in
/// vanilla plonk
pub(crate) fn build_f<F: PrimeField>(
    gates: &CustomizedGates,
    num_vars: usize,
    selector_mles: &[Arc<DenseMultilinearExtension<F>>],
    witness_mles: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<VirtualPolynomial<F>, HyperPlonkErrors> {
    // TODO: check that selector and witness lengths match what is in
    // the gate definition

    for selector_mle in selector_mles.iter() {
        if selector_mle.num_vars != num_vars {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "selector has different number of vars: {} vs {}",
                selector_mle.num_vars, num_vars
            )));
        }
    }

    for witness_mle in witness_mles.iter() {
        if witness_mle.num_vars != num_vars {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "selector has different number of vars: {} vs {}",
                witness_mle.num_vars, num_vars
            )));
        }
    }

    let mut res = VirtualPolynomial::<F>::new(num_vars);

    for (coeff, selector, witnesses) in gates.gates.iter() {
        let coeff_fr = if *coeff < 0 {
            -F::from(-*coeff as u64)
        } else {
            F::from(*coeff as u64)
        };
        let mut mle_list = vec![];
        if let Some(s) = *selector {
            mle_list.push(selector_mles[s].clone())
        }
        for &witness in witnesses.iter() {
            mle_list.push(witness_mles[witness].clone())
        }
        res.add_mle_list(mle_list, coeff_fr)?;
    }

    Ok(res)
}

pub(crate) fn eval_f<F: PrimeField>(
    gates: &CustomizedGates,
    selector_evals: &[F],
    witness_evals: &[F],
) -> Result<F, HyperPlonkErrors> {
    let mut res = F::zero();
    for (coeff, selector, witnesses) in gates.gates.iter() {
        let mut cur_value = if *coeff < 0 {
            -F::from(-*coeff as u64)
        } else {
            F::from(*coeff as u64)
        };
        cur_value *= match selector {
            Some(s) => selector_evals[*s],
            None => F::one(),
        };
        for &witness in witnesses.iter() {
            cur_value *= witness_evals[witness]
        }
        res += cur_value;
    }
    Ok(res)
}

pub(crate) fn build_f_product<F: PrimeField>(gates: &CustomizedGates) -> Vec<(F, Vec<usize>)> {
    let mut res = Vec::new();
    let num_witness_columns = gates.num_witness_columns();
    for (coeff, selector, witnesses) in gates.gates.iter() {
        let coeff_fr = if *coeff < 0 {
            -F::from(-*coeff as u64)
        } else {
            F::from(*coeff as u64)
        };
        let mut products = witnesses.clone();
        if let Some(s) = *selector {
            products.push(s + num_witness_columns);
        }
        res.push((coeff_fr, products));
    }
    res
}

pub(crate) fn build_f_raw<F: PrimeField>(
    gates: &CustomizedGates,
    num_vars: usize,
    selector_mles: &[Arc<DenseMultilinearExtension<F>>],
    witness_mles: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<VirtualPolynomial<F>, HyperPlonkErrors> {
    let aux_info = VPAuxInfo {
        max_degree: gates.degree(),
        num_variables: num_vars,
        phantom: PhantomData::default(),
    };

    if gates.num_selector_columns() != selector_mles.len() {
        return Err(HyperPlonkErrors::InvalidParameters(format!(
            "selector has different number of columns: {} vs {}",
            gates.num_selector_columns(),
            selector_mles.len()
        )));
    }

    if gates.num_witness_columns() != witness_mles.len() {
        return Err(HyperPlonkErrors::InvalidParameters(format!(
            "witness has different number of columns: {} vs {}",
            gates.num_witness_columns(),
            witness_mles.len()
        )));
    }

    for selector_mle in selector_mles.iter() {
        if selector_mle.num_vars != num_vars {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "selector has different number of vars: {} vs {}",
                selector_mle.num_vars, num_vars
            )));
        }
    }

    for witness_mle in witness_mles.iter() {
        if witness_mle.num_vars != num_vars {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "selector has different number of vars: {} vs {}",
                witness_mle.num_vars, num_vars
            )));
        }
    }

    let products = build_f_product(gates);

    Ok(VirtualPolynomial::new_from_raw(
        aux_info,
        products,
        witness_mles
            .iter()
            .chain(selector_mles.iter())
            .map(|x| x.clone())
            .collect(),
    ))
}

// check perm check subclaim:
// proof.witness_perm_check_eval ?= perm_check_sub_claim.expected_eval
// Q(x) := prod(x) - p1(x) * p2(x)
//     + alpha * frac(x) * g1(x) * ... * gk(x)
//     - alpha * f1(x) * ... * fk(x)
//
// where p1(x) = (1-x1) * frac(x2, ..., xn, 0)
//             + x1 * prod(x2, ..., xn, 0),
// and p2(x) = (1-x1) * frac(x2, ..., xn, 1)
//           + x1 * prod(x2, ..., xn, 1)
// and gi(x) = (wi(x) + beta * perms_i(x) + gamma)
// and fi(x) = (wi(x) + beta * s_id_i(x) + gamma)
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_perm_gate<F: PrimeField>(
    prod_evals: &[F],
    frac_evals: &[F],
    witness_perm_evals: &[F],
    id_evals: &[F],
    perm_evals: &[F],
    alpha: F,
    beta: F,
    gamma: F,
    x1: F,
) -> Result<F, HyperPlonkErrors> {
    let p1_eval = frac_evals[1] + x1 * (prod_evals[1] - frac_evals[1]);
    let p2_eval = frac_evals[2] + x1 * (prod_evals[2] - frac_evals[2]);
    let mut f_prod_eval = F::one();
    for (&w_eval, &id_eval) in witness_perm_evals.iter().zip(id_evals.iter()) {
        f_prod_eval *= w_eval + beta * id_eval + gamma;
    }
    let mut g_prod_eval = F::one();
    for (&w_eval, &p_eval) in witness_perm_evals.iter().zip(perm_evals.iter()) {
        g_prod_eval *= w_eval + beta * p_eval + gamma;
    }
    let res =
        prod_evals[0] - p1_eval * p2_eval + alpha * (frac_evals[0] * g_prod_eval - f_prod_eval);
    Ok(res)
}

// check distributed permutation check subclaim:
// proof.witness_perm_check_eval ?= perm_check_sub_claim.expected_eval
// Q(x) := alpha1 * prod_master(x) - alpha1 * p1_master(x) * p2_master(x)
//       + alpha0 * prod_worker(x) - alpha0 * p1_worker(x) * p2_worker(x)
//       + frac(x) * g1(x) * ... * gk(x) - f1(x) * ... * fk(x)
// where p1_master(x) = (1-x1) * prod_worker(x_2..t, 0, 1, ..., 1, 0) + x1 * prod_master(x_2..t, 0, x_t+1..n)
//       p2_master(x) = (1-x1) * prod_worker(x_2..t, 1, 0, ..., 0, 1) + x1 * prod_master(x_2..t, 1, x_t+1..n)
//       p1_worker(x) = (1-x_{t+1}) * frac(x_1..t, x_{t+2}..n, 0) + x_{t+1} * prod_worker(x_1..t, x_{t+2}..n, 0)
//       p2_worker(x) = (1-x_{t+1}) * frac(x_1..t, x_{t+2}..n, 1) + x_{t+1} * prod_worker(x_1..t, x_{t+2}..n, 1)
//       gi(x) = (wi(x) + beta * perms_i(x) + gamma)
//       fi(x) = (wi(x) + beta * s_id_i(x) + gamma)
//       t = log2(num_workers)
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_perm_gate_distributed<F: PrimeField>(
    prod_master_evals: &[F],
    prod_worker_evals: &[F],
    frac_evals: &[F],
    perm_evals: &[F],
    id_evals: &[F],
    witness_perm_evals: &[F],
    alpha0: F,
    alpha1: F,
    beta: F,
    gamma: F,
    x1: F,
    x_t_1: F,
) -> Result<F, HyperPlonkErrors> {
    let p1_master_eval = (F::one() - x1) * prod_worker_evals[0] + x1 * prod_master_evals[1];
    let p2_master_eval = (F::one() - x1) * prod_worker_evals[1] + x1 * prod_master_evals[2];

    let p1_worker_eval = (F::one() - x_t_1) * frac_evals[1] + x_t_1 * prod_worker_evals[3];
    let p2_worker_eval = (F::one() - x_t_1) * frac_evals[2] + x_t_1 * prod_worker_evals[4];

    let mut f = F::one();
    for (&w_eval, &id_eval) in witness_perm_evals.iter().zip(id_evals.iter()) {
        f *= w_eval + beta * id_eval + gamma;
    }
    let mut g = F::one();
    for (&w_eval, &perm_eval) in witness_perm_evals.iter().zip(perm_evals.iter()) {
        g *= w_eval + beta * perm_eval + gamma;
    }

    let res = alpha1 * (prod_master_evals[0] - p1_master_eval * p2_master_eval)
        + alpha0 * (prod_worker_evals[2] - p1_worker_eval * p2_worker_eval)
        + frac_evals[0] * g
        - f;

    Ok(res)
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::PrimeField;
    use ark_poly::MultilinearExtension;
    #[test]
    fn test_build_gate() -> Result<(), HyperPlonkErrors> {
        test_build_gate_helper::<Fr>()
    }

    fn test_build_gate_helper<F: PrimeField>() -> Result<(), HyperPlonkErrors> {
        let num_vars = 2;

        // ql = 3x1x2 + 2x2 whose evaluations are
        // 0, 0 |-> 0
        // 0, 1 |-> 2
        // 1, 0 |-> 0
        // 1, 1 |-> 5
        let ql_eval = vec![F::zero(), F::from(2u64), F::zero(), F::from(5u64)];
        let ql = Arc::new(DenseMultilinearExtension::from_evaluations_vec(2, ql_eval));

        // W1 = x1x2 + x1 whose evaluations are
        // 0, 0 |-> 0
        // 0, 1 |-> 0
        // 1, 0 |-> 1
        // 1, 1 |-> 2
        let w_eval = vec![F::zero(), F::zero(), F::from(1u64), F::from(2u64)];
        let w1 = Arc::new(DenseMultilinearExtension::from_evaluations_vec(2, w_eval));

        // W2 = x1 + x2 whose evaluations are
        // 0, 0 |-> 0
        // 0, 1 |-> 1
        // 1, 0 |-> 1
        // 1, 1 |-> 2
        let w_eval = vec![F::zero(), F::one(), F::from(1u64), F::from(2u64)];
        let w2 = Arc::new(DenseMultilinearExtension::from_evaluations_vec(2, w_eval));

        // Example:
        //     q_L(X) * W_1(X)^5 - W_2(X)
        // is represented as
        // vec![
        //     ( 1,    Some(id_qL),    vec![id_W1, id_W1, id_W1, id_W1, id_W1]),
        //     (-1,    None,           vec![id_W2])
        // ]
        let gates = CustomizedGates {
            gates: vec![(1, Some(0), vec![0, 0, 0, 0, 0]), (-1, None, vec![1])],
        };
        let f = build_f(&gates, num_vars, &[ql.clone()], &[w1.clone(), w2.clone()])?;

        // Sanity check on build_f
        // f(0, 0) = 0
        assert_eq!(f.evaluate(&[F::zero(), F::zero()])?, F::zero());
        // f(0, 1) = 2 * 0^5 + (-1) * 1 = -1
        assert_eq!(f.evaluate(&[F::zero(), F::one()])?, -F::one());
        // f(1, 0) = 0 * 1^5 + (-1) * 1 = -1
        assert_eq!(f.evaluate(&[F::one(), F::zero()])?, -F::one());
        // f(1, 1) = 5 * 2^5 + (-1) * 2 = 158
        assert_eq!(f.evaluate(&[F::one(), F::one()])?, F::from(158u64));

        // test eval_f
        {
            let point = [F::zero(), F::zero()];
            let selector_evals = ql.evaluate(&point).unwrap();
            let witness_evals = [w1.evaluate(&point).unwrap(), w2.evaluate(&point).unwrap()];
            let eval_f = eval_f(&gates, &[selector_evals], &witness_evals)?;
            // f(0, 0) = 0
            assert_eq!(eval_f, F::zero());
        }
        {
            let point = [F::zero(), F::one()];
            let selector_evals = ql.evaluate(&point).unwrap();
            let witness_evals = [w1.evaluate(&point).unwrap(), w2.evaluate(&point).unwrap()];
            let eval_f = eval_f(&gates, &[selector_evals], &witness_evals)?;
            // f(0, 1) = 2 * 0^5 + (-1) * 1 = -1
            assert_eq!(eval_f, -F::one());
        }
        {
            let point = [F::one(), F::zero()];
            let selector_evals = ql.evaluate(&point).unwrap();
            let witness_evals = [w1.evaluate(&point).unwrap(), w2.evaluate(&point).unwrap()];
            let eval_f = eval_f(&gates, &[selector_evals], &witness_evals)?;
            // f(1, 0) = 0 * 1^5 + (-1) * 1 = -1
            assert_eq!(eval_f, -F::one());
        }
        {
            let point = [F::one(), F::one()];
            let selector_evals = ql.evaluate(&point).unwrap();
            let witness_evals = [w1.evaluate(&point).unwrap(), w2.evaluate(&point).unwrap()];
            let eval_f = eval_f(&gates, &[selector_evals], &witness_evals)?;
            // f(1, 1) = 5 * 2^5 + (-1) * 2 = 158
            assert_eq!(eval_f, F::from(158u64));
        }
        Ok(())
    }
}
