use std::{marker::PhantomData, sync::Arc};

use arithmetic::{identity_permutation, split_into_chunks, transpose, VPAuxInfo};
use ark_ec::pairing::Pairing;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{end_timer, log2, start_timer, One, Zero};
use rayon::iter::IntoParallelRefIterator;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use subroutines::{
    pcs::PolynomialCommitmentSchemeDistributed, BatchProof, Commitment, MultilinearProverParam,
    PermutationCheckDistributed, PolyIOP, WorkerProverChannel, ZeroCheck, ZeroCheckDistributed,
};
use transcript::IOPTranscript;

use crate::{
    errors::HyperPlonkErrors,
    prelude::WitnessColumn,
    structs::{
        HyperPlonkIndex, HyperPlonkProofDistributed, HyperPlonkProvingKeyMaster,
        HyperPlonkProvingKeyWorker, HyperPlonkVerifyingKey,
    },
    utils::{
        build_f_product, build_f_raw, eval_f, eval_perm_gate_distributed, prover_sanity_check,
        PcsAccumulatorMaster, PcsAccumulatorWorker,
    },
    HyperPlonkSNARKDistributed,
};

impl<E, PCS> HyperPlonkSNARKDistributed<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
        BatchProof = BatchProof<E, PCS>,
        MasterPolynomialHandle = usize,
        WorkerPolynomialHandle = Arc<DenseMultilinearExtension<E::ScalarField>>,
        ProverParam = MultilinearProverParam<E>,
        MasterProverParam = MultilinearProverParam<E>,
    >,
{
    type Index = HyperPlonkIndex<E::ScalarField>;
    type ProvingKeyMaster = HyperPlonkProvingKeyMaster<E, PCS>;
    type ProvingKeyWorker = HyperPlonkProvingKeyWorker<E, PCS>;
    type VerifyingKey = HyperPlonkVerifyingKey<E, PCS>;
    type Proof = HyperPlonkProofDistributed<E, Self, PCS>;

    fn preprocess(
        index: &Self::Index,
        log_num_worker: usize,
        pcs_srs: &PCS::SRS,
    ) -> Result<
        (
            (Self::ProvingKeyMaster, Vec<Self::ProvingKeyWorker>),
            Self::VerifyingKey,
        ),
        HyperPlonkErrors,
    > {
        let num_vars = index.num_variables();
        let supported_ml_degree = num_vars;

        let (pcs_prover_param, pcs_verifier_param) =
            PCS::trim(pcs_srs, None, Some(supported_ml_degree))?;

        let mut permutation_oracles = Vec::new();
        let mut perm_comms = Vec::new();
        let chunk_size = 1 << num_vars;
        for i in 0..index.num_witness_columns() {
            let perm_oracle = Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars,
                &index.permutation[i * chunk_size..(i + 1) * chunk_size],
            ));
            let perm_comm = PCS::commit(&pcs_prover_param, &perm_oracle)?;
            permutation_oracles.push(perm_oracle);
            perm_comms.push(perm_comm);
        }

        let permutation_oracles_distributed = permutation_oracles
            .par_iter()
            .map(|perm| split_into_chunks(perm, log_num_worker))
            .collect::<Vec<_>>();
        let permutation_oracles_distributed = transpose(permutation_oracles_distributed);

        let selector_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = index
            .selectors
            .iter()
            .map(|s| Arc::new(DenseMultilinearExtension::from(s)))
            .collect();

        let selector_commitments = selector_oracles
            .par_iter()
            .map(|poly| PCS::commit(&pcs_prover_param, poly))
            .collect::<Result<Vec<_>, _>>()?;

        let selector_oracles_distributed = selector_oracles
            .par_iter()
            .map(|selector| split_into_chunks(selector, log_num_worker))
            .collect::<Vec<_>>();
        let selector_oracles_distributed = transpose(selector_oracles_distributed);

        let identity_permutation = identity_permutation(num_vars, index.num_witness_columns());
        let identity_oracles = identity_permutation
            .chunks(chunk_size)
            .map(|chunk| {
                Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                    num_vars, chunk,
                ))
            })
            .collect::<Vec<_>>();
        let identity_oracles_distributed = identity_oracles
            .par_iter()
            .map(|identity| split_into_chunks(identity, log_num_worker))
            .collect::<Vec<_>>();
        let identity_oracles_distributed = transpose(identity_oracles_distributed);

        let (pcs_param_master, pcs_param_worker) =
            PCS::prover_param_distributed(pcs_prover_param, log_num_worker)?;
        Ok((
            (
                HyperPlonkProvingKeyMaster {
                    params: index.params.clone(),
                    selector_commitments: selector_commitments.clone(),
                    permutation_commitments: perm_comms.clone(),
                    pcs_param: pcs_param_master,
                },
                pcs_param_worker
                    .into_iter()
                    .zip(selector_oracles_distributed.into_iter())
                    .zip(
                        permutation_oracles_distributed
                            .into_iter()
                            .zip(identity_oracles_distributed.into_iter()),
                    )
                    .map(
                        |((pcs_param, selector_oracles), (perm_oracles, id_oracles))| {
                            HyperPlonkProvingKeyWorker {
                                params: index.params.clone(),
                                selector_oracles,
                                perm_oracles,
                                id_oracles,
                                pcs_param,
                            }
                        },
                    )
                    .collect(),
            ),
            HyperPlonkVerifyingKey {
                params: index.params.clone(),
                pcs_param: pcs_verifier_param,
                selector_commitments,
                perm_commitments: perm_comms,
            },
        ))
    }

    fn prove_master(
        pk: &Self::ProvingKeyMaster,
        pub_input: &[E::ScalarField],
        witnesses: &[WitnessColumn<E::ScalarField>],
        log_num_workers: usize,
        master_channel: &mut impl subroutines::MasterProverChannel,
    ) -> Result<Self::Proof, HyperPlonkErrors> {
        let start = start_timer!(|| "Cirrus; master");

        let mut transcript = IOPTranscript::<E::ScalarField>::new(b"cirrus");

        prover_sanity_check(&pk.params, pub_input, witnesses)?;
        let num_vars = pk.params.num_variables();
        assert!(
            log_num_workers == master_channel.log_num_workers(),
            "log_num_workers mismatch with the master channel"
        );
        let log_num_workers = master_channel.log_num_workers();
        let ell = log2(pk.params.num_pub_input) as usize;

        let mut pcs_acc = PcsAccumulatorMaster::<E, PCS>::new(num_vars);

        // =======================================================================
        // 1. Master prover distributes the witness polynomials to workers and
        //    commits to them
        // =======================================================================
        let step = start_timer!(|| "distribute and commit witnesses; master");

        assert!(
            num_vars >= log_num_workers + 1,
            "num_vars must be greater than log_num_workers"
        );
        let witnesses_distribution = witnesses
            .iter()
            .map(|w| {
                w.0.chunks(1 << (num_vars - log_num_workers))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let witnesses_distribution = transpose(witnesses_distribution);

        master_channel.send_different(witnesses_distribution)?;
        let witness_commits = (0..witnesses.len())
            .map(|_| PCS::commit_distributed_master(&pk.pcs_param, &num_vars, master_channel))
            .collect::<Result<Vec<_>, _>>()?;
        for w_com in witness_commits.iter() {
            transcript.append_serializable_element(b"w", w_com)?;
        }

        end_timer!(step);

        // =======================================================================
        // 2. All provers run distributed zero check on
        //          f(q_0(x),...q_l(x), w_0(x),...w_d(x))
        // where f is the gate function
        // =======================================================================

        let step = start_timer!(|| "Distributed zero check on f; master");

        let aux_info = VPAuxInfo {
            max_degree: pk.params.gate_func.degree(),
            num_variables: num_vars,
            phantom: PhantomData::default(),
        };
        let products = build_f_product(&pk.params.gate_func);
        let zero_check_proof = <Self as ZeroCheckDistributed<_>>::prove_master(
            &aux_info,
            &products,
            log_num_workers,
            &mut transcript,
            master_channel,
        )?;

        end_timer!(step);

        // =======================================================================
        // 3. All provers run distributed permutation check on the witness
        // polynomials and the permutation oracles.
        // =======================================================================

        let step =
            start_timer!(|| "Distributed permutation check on witnesses and permutations; master");

        let (perm_check_proof, prod_master) =
            <Self as PermutationCheckDistributed<E, PCS>>::prove_master(
                &pk.pcs_param,
                witnesses.len(),
                num_vars,
                &mut transcript,
                master_channel,
            )?;

        master_channel.send_different(prod_master.evaluations.clone())?;

        end_timer!(step);

        // =======================================================================
        // 4. generate pcs batch proof
        // =======================================================================

        let step = start_timer!(|| "generate pcs batch proof; master");

        let perm_check_point = perm_check_proof.zero_check_proof.point.clone();
        // prod_master 's points
        let point1 = perm_check_point.clone();
        let point2 = [
            &perm_check_point[0..(num_vars - log_num_workers)],
            &[E::ScalarField::zero()],
            &perm_check_point[(num_vars - log_num_workers)..(num_vars - 1)],
        ]
        .concat();
        let point3 = [
            &perm_check_point[0..(num_vars - log_num_workers)],
            &[E::ScalarField::one()],
            &perm_check_point[(num_vars - log_num_workers)..(num_vars - 1)],
        ]
        .concat();
        let mut point4 = vec![E::ScalarField::one(); num_vars];
        point4[num_vars - log_num_workers] = E::ScalarField::zero();
        pcs_acc.insert_point(point1);
        pcs_acc.insert_point(point2);
        pcs_acc.insert_point(point3);
        pcs_acc.insert_point(point4);

        // prod_worker 's points
        let point1 = [
            &[E::ScalarField::zero()],
            &vec![E::ScalarField::one(); num_vars - 1 - log_num_workers][..],
            &[E::ScalarField::zero()],
            &perm_check_point[(num_vars - log_num_workers)..(num_vars - 1)],
        ]
        .concat();

        let point2 = [
            &[E::ScalarField::zero()],
            &vec![E::ScalarField::one(); num_vars - log_num_workers][..],
            &perm_check_point[(num_vars - log_num_workers)..(num_vars - 1)],
        ]
        .concat();

        let point4 = [
            &[E::ScalarField::zero()],
            &perm_check_point[0..(num_vars - log_num_workers - 1)],
            &perm_check_point[(num_vars - log_num_workers)..],
        ]
        .concat();

        let point5 = [
            &[E::ScalarField::one()],
            &perm_check_point[0..(num_vars - log_num_workers - 1)],
            &perm_check_point[(num_vars - log_num_workers)..],
        ]
        .concat();

        pcs_acc.insert_point(point1);
        pcs_acc.insert_point(point2);
        pcs_acc.insert_point(perm_check_point.clone());
        pcs_acc.insert_point(point4.clone());
        pcs_acc.insert_point(point5.clone());

        // frac 's points
        pcs_acc.insert_point(perm_check_point.clone());
        pcs_acc.insert_point(point4);
        pcs_acc.insert_point(point5);

        // perm 's points
        for _ in 0..pk.params.gate_func.num_witness_columns() {
            pcs_acc.insert_point(perm_check_point.clone());
        }

        // witness 's points
        for _ in 0..pk.params.gate_func.num_witness_columns() {
            pcs_acc.insert_point(perm_check_point.clone());
        }

        for _ in 0..pk.params.gate_func.num_witness_columns() {
            pcs_acc.insert_point(zero_check_proof.point.clone());
        }

        // selector 's points
        for _ in 0..pk.params.gate_func.num_selector_columns() {
            pcs_acc.insert_point(zero_check_proof.point.clone());
        }

        let batch_openings = pcs_acc.multi_open(&pk.pcs_param, &mut transcript, master_channel)?;

        end_timer!(step);

        end_timer!(start);
        Ok(HyperPlonkProofDistributed {
            witness_commits,
            batch_openings,
            zero_check_proof,
            perm_check_proof,
        })
    }

    fn prove_worker(
        pk: &Self::ProvingKeyWorker,
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(), HyperPlonkErrors> {
        let start = start_timer!(|| "Cirrus; worker");

        let num_vars = pk.selector_oracles[0].num_vars();
        let mut pcs_acc = PcsAccumulatorWorker::<E, PCS>::new(num_vars);

        // =======================================================================
        // 1. Worker prover receives the witness polynomials and commits to them
        // =======================================================================
        let step = start_timer!(|| "distribute and commit witnesses; master");

        let witnesses: Vec<Vec<E::ScalarField>> = worker_channel.recv()?;
        let witness_polys = witnesses
            .into_iter()
            .map(|w| Arc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, w)))
            .collect::<Vec<_>>();

        witness_polys
            .iter()
            .map(|w| PCS::commit_distributed_worker(&pk.pcs_param, w, worker_channel))
            .collect::<Result<Vec<_>, _>>()?;

        end_timer!(step);

        // =======================================================================
        // 2. Worker prover runs distributed zero check on
        //          f(q_0(x),...q_l(x), w_0(x),...w_d(x))
        // where f is the gate function
        // =======================================================================
        let step = start_timer!(|| "Distributed zero check on f; worker");

        let fx = build_f_raw(
            &pk.params.gate_func,
            num_vars,
            &pk.selector_oracles,
            &witness_polys,
        )?;

        <Self as ZeroCheckDistributed<E::ScalarField>>::prove_worker(&fx, worker_channel)?;

        end_timer!(step);

        // =======================================================================
        // 3. Worker prover runs distributed permutation check on the witness
        // polynomials and the permutation oracles.
        // =======================================================================
        let step =
            start_timer!(|| "Distributed permutation check on witnesses and permutations; worker");

        let (prod_worker, frac) = <Self as PermutationCheckDistributed<E, PCS>>::prove_worker(
            &pk.pcs_param,
            &witness_polys,
            &witness_polys,
            &pk.id_oracles,
            &pk.perm_oracles,
            worker_channel,
        )?;

        end_timer!(step);

        // =======================================================================
        // 4. generate pcs batch proof. Worker provers first insert their part of
        // polynomials to the accumulator, and batch opening them with points from
        // the master prover.
        // =======================================================================

        let step = start_timer!(|| "generate pcs batch proof; worker");

        let prod_master = {
            let f: E::ScalarField = worker_channel.recv()?;
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                vec![f; 1 << num_vars],
            ))
        };

        pcs_acc.insert_poly(&prod_master);
        pcs_acc.insert_poly(&prod_master);
        pcs_acc.insert_poly(&prod_master);
        pcs_acc.insert_poly(&prod_master);

        pcs_acc.insert_poly(&prod_worker);
        pcs_acc.insert_poly(&prod_worker);
        pcs_acc.insert_poly(&prod_worker);
        pcs_acc.insert_poly(&prod_worker);
        pcs_acc.insert_poly(&prod_worker);

        pcs_acc.insert_poly(&frac);
        pcs_acc.insert_poly(&frac);
        pcs_acc.insert_poly(&frac);

        for perm in pk.perm_oracles.iter() {
            pcs_acc.insert_poly(perm);
        }

        for w_poly in witness_polys.iter() {
            pcs_acc.insert_poly(w_poly);
        }

        for w_poly in witness_polys.iter() {
            pcs_acc.insert_poly(w_poly);
        }

        for selector in pk.selector_oracles.iter() {
            pcs_acc.insert_poly(selector);
        }

        pcs_acc.multi_open(&pk.pcs_param, worker_channel)?;

        end_timer!(step);

        // =======================================================================
        // Worker prover ends
        // =======================================================================

        end_timer!(start);
        Ok(())
    }

    fn verify(
        vk: &Self::VerifyingKey,
        pub_input: &[<E as Pairing>::ScalarField],
        proof: &Self::Proof,
    ) -> Result<bool, HyperPlonkErrors> {
        let num_vars = vk.params.num_variables();
        let log_num_workers = proof.perm_check_proof.log_num_workers;
        let num_pub_input = vk.params.num_pub_input;
        let num_witness_columns = vk.params.gate_func.num_witness_columns();
        let num_selector_columns = vk.params.gate_func.num_selector_columns();

        let mut transcript = IOPTranscript::<E::ScalarField>::new(b"cirrus");
        for w_com in proof.witness_commits.iter() {
            transcript.append_serializable_element(b"w", w_com)?;
        }

        // evaluations

        let prod_master_evals = &proof.batch_openings.f_i_eval_at_point_i[0..4];
        let prod_worker_evals = &proof.batch_openings.f_i_eval_at_point_i[4..9];
        let frac_evals = &proof.batch_openings.f_i_eval_at_point_i[9..12];
        let perm_evals = &proof.batch_openings.f_i_eval_at_point_i[12..12 + num_witness_columns];
        let witness_perm_evals = &proof.batch_openings.f_i_eval_at_point_i
            [12 + num_witness_columns..12 + 2 * num_witness_columns];
        let witness_gate_evals = &proof.batch_openings.f_i_eval_at_point_i
            [12 + 2 * num_witness_columns..12 + 3 * num_witness_columns];
        let selector_evals = &proof.batch_openings.f_i_eval_at_point_i
            [12 + 3 * num_witness_columns..12 + 3 * num_witness_columns + num_selector_columns];

        // zero check

        let aux_info = VPAuxInfo {
            max_degree: vk.params.gate_func.degree(),
            num_variables: num_vars,
            phantom: PhantomData::default(),
        };
        let zero_check_subclaim = <Self as ZeroCheck<E::ScalarField>>::verify(
            &proof.zero_check_proof,
            &aux_info,
            &mut transcript,
        )?;

        let zero_check_point = zero_check_subclaim.point;

        // check zero check subclaim
        let f_eval = eval_f(&vk.params.gate_func, selector_evals, witness_gate_evals)?;
        if f_eval != zero_check_subclaim.expected_evaluation {
            return Err(HyperPlonkErrors::InvalidProof(
                "zero check evaluation failed".to_string(),
            ));
        }

        // permutation check

        let aux_info = VPAuxInfo {
            max_degree: num_witness_columns + 1,
            num_variables: num_vars,
            phantom: PhantomData::default(),
        };
        let perm_check_subclaim = <Self as PermutationCheckDistributed<E, PCS>>::verify(
            &proof.perm_check_proof,
            &aux_info,
            &mut transcript,
        )?;

        let perm_check_point = perm_check_subclaim
            .product_check_sub_claim
            .zero_check_sub_claim
            .point;

        let expected_eval = perm_check_subclaim
            .product_check_sub_claim
            .zero_check_sub_claim
            .expected_evaluation;

        let (alpha1, alpha2) = perm_check_subclaim.product_check_sub_claim.alpha;
        let (beta, gamma) = perm_check_subclaim.challenges;

        let id1 = {
            let mut base = E::ScalarField::one();
            let mut res = E::ScalarField::zero();
            for x in perm_check_point.iter() {
                res += base * x;
                base += base;
            }
            res
        };

        let id_evals = (0..num_witness_columns)
            .map(|i| E::ScalarField::from((i * (1 << num_vars)) as u128) + id1)
            .collect::<Vec<_>>();

        if expected_eval
            != eval_perm_gate_distributed(
                prod_master_evals,
                prod_worker_evals,
                frac_evals,
                perm_evals,
                &id_evals[..],
                witness_perm_evals,
                alpha1,
                alpha2,
                beta,
                gamma,
                perm_check_point[num_vars - 1],
                perm_check_point[num_vars - log_num_workers - 1],
            )?
        {
            return Err(HyperPlonkErrors::InvalidVerifier(
                "evaluation of identity constraint(permutation check) failed".to_string(),
            ));
        }

        let mut comms = vec![];
        let mut points = vec![];

        let point1 = perm_check_point.clone();
        let point2 = [
            &perm_check_point[0..(num_vars - log_num_workers)],
            &[E::ScalarField::zero()],
            &perm_check_point[(num_vars - log_num_workers)..(num_vars - 1)],
        ]
        .concat();
        let point3 = [
            &perm_check_point[0..(num_vars - log_num_workers)],
            &[E::ScalarField::one()],
            &perm_check_point[(num_vars - log_num_workers)..(num_vars - 1)],
        ]
        .concat();
        let mut point4 = vec![E::ScalarField::one(); num_vars];
        point4[num_vars - log_num_workers] = E::ScalarField::zero();

        comms.push(proof.perm_check_proof.prod_master_comm);
        comms.push(proof.perm_check_proof.prod_master_comm);
        comms.push(proof.perm_check_proof.prod_master_comm);
        comms.push(proof.perm_check_proof.prod_master_comm);
        points.push(point1);
        points.push(point2);
        points.push(point3);
        points.push(point4);

        let point1 = [
            &[E::ScalarField::zero()],
            &vec![E::ScalarField::one(); num_vars - 1 - log_num_workers][..],
            &[E::ScalarField::zero()],
            &perm_check_point[(num_vars - log_num_workers)..(num_vars - 1)],
        ]
        .concat();

        let point2 = [
            &[E::ScalarField::zero()],
            &vec![E::ScalarField::one(); num_vars - log_num_workers][..],
            &perm_check_point[(num_vars - log_num_workers)..(num_vars - 1)],
        ]
        .concat();

        let point4 = [
            &[E::ScalarField::zero()],
            &perm_check_point[0..(num_vars - log_num_workers - 1)],
            &perm_check_point[(num_vars - log_num_workers)..],
        ]
        .concat();

        let point5 = [
            &[E::ScalarField::one()],
            &perm_check_point[0..(num_vars - log_num_workers - 1)],
            &perm_check_point[(num_vars - log_num_workers)..],
        ]
        .concat();

        comms.push(proof.perm_check_proof.prod_worker_comm);
        comms.push(proof.perm_check_proof.prod_worker_comm);
        comms.push(proof.perm_check_proof.prod_worker_comm);
        comms.push(proof.perm_check_proof.prod_worker_comm);
        comms.push(proof.perm_check_proof.prod_worker_comm);

        points.push(point1);
        points.push(point2);
        points.push(perm_check_point.clone());
        points.push(point4.clone());
        points.push(point5.clone());

        comms.push(proof.perm_check_proof.frac_comm);
        comms.push(proof.perm_check_proof.frac_comm);
        comms.push(proof.perm_check_proof.frac_comm);

        points.push(perm_check_point.clone());
        points.push(point4);
        points.push(point5);

        for &comm in vk.perm_commitments.iter() {
            comms.push(comm);
            points.push(perm_check_point.clone());
        }

        for &comm in proof.witness_commits.iter() {
            comms.push(comm);
            points.push(perm_check_point.clone());
        }

        for &comm in proof.witness_commits.iter() {
            comms.push(comm);
            points.push(zero_check_point.clone());
        }

        for &comm in vk.selector_commitments.iter() {
            comms.push(comm);
            points.push(zero_check_point.clone());
        }

        let res = PCS::batch_verify(
            &vk.pcs_param,
            &comms,
            &points,
            &proof.batch_openings,
            &mut transcript,
        )?;

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use std::thread::spawn;

    use ark_bls12_381::Bls12_381;
    use ark_std::{test_rng, One};
    use subroutines::{new_master_worker_channels, MultilinearKzgPCS, PolynomialCommitmentScheme};

    use crate::{
        prelude::{CustomizedGates, SelectorColumn},
        structs::HyperPlonkParams,
    };

    use super::*;

    #[test]
    fn test() {
        let test = &[
            [1, 2, 4, 5, 6, 7],
            [9, 20, 1, 3, 2, 3],
            [9, 20, 1, 3, 10, 3],
        ];
        dbg!(test
            .iter()
            .map(|t| t.chunks(2).collect::<Vec<_>>())
            .collect::<Vec<_>>());
    }

    #[test]
    fn test_hyperplonk_distributed() -> Result<(), HyperPlonkErrors> {
        test_hyperplonk_distributed_helper::<Bls12_381>()
    }

    fn test_hyperplonk_distributed_helper<E>() -> Result<(), HyperPlonkErrors>
    where
        E: Pairing,
    {
        let mut rng = test_rng();
        let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 16)?;

        let nv = 3 as usize;
        let num_constraints = 1 << nv;
        let num_pub_input = 4;
        let log_num_workers = 1;

        // q0 * w0^3 + (-1) * w1 = 0
        let gate_func = CustomizedGates {
            gates: vec![(1, Some(0), vec![0; 3]), (-1, None, vec![1])],
        };
        let params = HyperPlonkParams {
            num_constraints,
            num_pub_input,
            gate_func,
        };

        // w1: 0, 1, 2, 3, 0, 1, 2, 3
        let w1 = WitnessColumn(
            (0..num_constraints)
                .map(|i| E::ScalarField::from((i % 4) as u128))
                .collect(),
        );
        // w2: 0, 1, 8, 27, 0, 1, 8, 27
        let w2 = WitnessColumn(
            (0..num_constraints)
                .map(|i| i % 4)
                .map(|i| E::ScalarField::from((i * i * i) as u128))
                .collect(),
        );
        let permutation = vec![
            E::ScalarField::from(8u128),  // 0: 0
            E::ScalarField::one(),        // 1: 1
            E::ScalarField::from(2u128),  // 2: 2
            E::ScalarField::from(3u128),  // 3: 3
            E::ScalarField::zero(),       // 4: 0
            E::ScalarField::from(5u128),  // 5: 1
            E::ScalarField::from(6u128),  // 6: 2
            E::ScalarField::from(7u128),  // 7: 3
            E::ScalarField::from(4u128),  // 8: 0
            E::ScalarField::from(9u128),  // 9: 1
            E::ScalarField::from(10u128), // 10: 8
            E::ScalarField::from(15u128), // 11: 27
            E::ScalarField::from(12u128), // 12: 0
            E::ScalarField::from(13u128), // 13: 1
            E::ScalarField::from(14u128), // 14: 8
            E::ScalarField::from(11u128), // 15: 27
        ];
        let q = SelectorColumn(vec![E::ScalarField::one(); num_constraints]);
        let index = HyperPlonkIndex {
            params: params.clone(),
            permutation,
            selectors: vec![q],
        };

        // let (mut master_channel, worker_channels) = new_master_worker_thread_channels(log_num_workers);
        let (mut master_channel, worker_channels) =
            new_master_worker_channels(true, log_num_workers, "127.0.0.1:0");

        let ((pk_master, pk_workers), vk) =
            PolyIOP::<E::ScalarField>::preprocess(&index, log_num_workers, &pcs_srs)?;

        let worker_handles = pk_workers
            .into_iter()
            .zip(worker_channels.into_iter())
            .map(|(pk, mut channel)| {
                spawn(move || {
                    <PolyIOP<E::ScalarField> as HyperPlonkSNARKDistributed<
                        E,
                        MultilinearKzgPCS<E>,
                    >>::prove_worker(&pk, &mut channel)
                })
            })
            .collect::<Vec<_>>();

        let proof = <PolyIOP<E::ScalarField> as HyperPlonkSNARKDistributed<
            E,
            MultilinearKzgPCS<E>,
        >>::prove_master(
            &pk_master,
            w1.clone().0[..num_pub_input].as_ref(),
            &[w1.clone(), w2],
            log_num_workers,
            &mut master_channel,
        )?;

        for handle in worker_handles {
            handle.join().unwrap()?;
        }

        assert!(<PolyIOP<E::ScalarField> as HyperPlonkSNARKDistributed<
            E,
            MultilinearKzgPCS<E>,
        >>::verify(&vk, &w1.0[..num_pub_input], &proof)?);

        Ok(())
    }
}
