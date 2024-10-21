// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements the sum check protocol.

use crate::{poly_iop::{
    errors::PolyIOPErrors,
    structs::{IOPProof, IOPProverMessage, IOPProverState, IOPVerifierState},
    PolyIOP,
}, MasterProverChannel, WorkerProverChannel};
use arithmetic::{VPAuxInfo, VirtualPolynomial};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{fmt::Debug, sync::Arc};
use transcript::IOPTranscript;

mod prover;
mod verifier;

/// Trait for doing sum check protocols.
pub trait SumCheck<F: PrimeField> {
    type VirtualPolynomial;
    type VPAuxInfo;
    type MultilinearExtension;

    type SumCheckProof: Clone + Debug + Default + PartialEq;
    type Transcript;
    type SumCheckSubClaim: Clone + Debug + Default + PartialEq;

    /// Extract sum from the proof
    fn extract_sum(proof: &Self::SumCheckProof) -> F;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a SumCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// SumCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// Generate proof of the sum of polynomial over {0,1}^`num_vars`
    ///
    /// The polynomial is represented in the form of a VirtualPolynomial.
    fn prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors>;

    /// Verify the claimed sum using the proof
    fn verify(
        sum: F,
        proof: &Self::SumCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors>;
}

/// Distributed version of sum check IOP. Each worker prover holds
/// a part of the multilinear extension. In the first phase, the master
/// prover sends challenges to the worker provers and collects the prover
/// messages. In the second phase, the master prover generates the proof
/// for the second phase just as the original sum check protocol.
/// The two parts are combined to form a complete zero check proof.
///
/// Therefore, we use the same proof type and verifier.
pub trait SumCheckDistributed<F: PrimeField>: SumCheck<F> {
    /// Master prover protocol of the distributed sum check for proving
    /// the sum of the polynomial over {0,1}^`num_vars`.
    ///
    /// Master prover holds the metadata of the polynomial, i.e., the
    /// `aux_info` and the `products`, which is defined in `VirtualPolynomial`.
    ///
    /// `master_channel` is the communication channel between the master
    /// prover and the worker provers.
    ///
    /// The `log_num_workers` is the log number of worker provers.
    /// The number of worker provers is `2^log_num_workers`. It also
    /// specifies the number of variables that master proves hold.
    fn prove_master(
        poly_aux_info: &Self::VPAuxInfo,
        poly_products: &Vec<(F, Vec<usize>)>,
        log_num_workers: usize,
        transcript: &mut Self::Transcript,
        master_channel: &impl MasterProverChannel,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors>;

    /// Worker prover protocol of the distributed sum check for proving
    /// the sum of the polynomial over {0,1}^`num_vars`.
    ///
    /// Worker prover holds a part of the multilinear extension of the
    /// `VirtualPolynomial`, and works as a sumcheck prover for the part.
    /// It receives challenges from the prover through the `worker_channel`.
    fn prove_worker(
        poly: &Self::VirtualPolynomial,
        worker_channel: &impl WorkerProverChannel,
    ) -> Result<(), PolyIOPErrors>;
}

/// Trait for sum check protocol prover side APIs.
pub trait SumCheckProver<F: PrimeField>
where
    Self: Sized,
{
    type VirtualPolynomial;
    type ProverMessage;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(polynomial: &Self::VirtualPolynomial) -> Result<Self, PolyIOPErrors>;

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &mut self,
        challenge: &Option<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors>;
}

/// Trait for sum check protocol verifier side APIs.
pub trait SumCheckVerifier<F: PrimeField> {
    type VPAuxInfo;
    type ProverMessage;
    type Challenge;
    type Transcript;
    type SumCheckSubClaim;

    /// Initialize the verifier's state.
    fn verifier_init(index_info: &Self::VPAuxInfo) -> Self;

    /// Run verifier for the current round, given a prover message.
    ///
    /// Note that `verify_round_and_update_state` only samples and stores
    /// challenges; and update the verifier's state accordingly. The actual
    /// verifications are deferred (in batch) to `check_and_generate_subclaim`
    /// at the last step.
    fn verify_round_and_update_state(
        &mut self,
        prover_msg: &Self::ProverMessage,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::Challenge, PolyIOPErrors>;

    /// This function verifies the deferred checks in the interactive version of
    /// the protocol; and generate the subclaim. Returns an error if the
    /// proof failed to verify.
    ///
    /// If the asserted sum is correct, then the multilinear polynomial
    /// evaluated at `subclaim.point` will be `subclaim.expected_evaluation`.
    /// Otherwise, it is highly unlikely that those two will be equal.
    /// Larger field size guarantees smaller soundness error.
    fn check_and_generate_subclaim(
        &self,
        asserted_sum: &F,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors>;
}

/// A SumCheckSubClaim is a claim generated by the verifier at the end of
/// verification when it is convinced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SumCheckSubClaim<F: PrimeField> {
    /// the multi-dimensional point that this multilinear extension is evaluated
    /// to
    pub point: Vec<F>,
    /// the expected evaluation
    pub expected_evaluation: F,
}

impl<F: PrimeField> SumCheck<F> for PolyIOP<F> {
    type SumCheckProof = IOPProof<F>;
    type VirtualPolynomial = VirtualPolynomial<F>;
    type VPAuxInfo = VPAuxInfo<F>;
    type MultilinearExtension = Arc<DenseMultilinearExtension<F>>;
    type SumCheckSubClaim = SumCheckSubClaim<F>;
    type Transcript = IOPTranscript<F>;

    fn extract_sum(proof: &Self::SumCheckProof) -> F {
        let start = start_timer!(|| "extract sum");
        let res = proof.proofs[0].evaluations[0] + proof.proofs[0].evaluations[1];
        end_timer!(start);
        res
    }

    fn init_transcript() -> Self::Transcript {
        let start = start_timer!(|| "init transcript");
        let res = IOPTranscript::<F>::new(b"Initializing SumCheck transcript");
        end_timer!(start);
        res
    }

    fn prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prove");

        transcript.append_serializable_element(b"aux info", &poly.aux_info)?;

        let mut prover_state = IOPProverState::prover_init(poly)?;
        let mut challenge = None;
        let mut prover_msgs = Vec::with_capacity(poly.aux_info.num_variables);
        for _ in 0..poly.aux_info.num_variables {
            let prover_msg =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            prover_msgs.push(prover_msg);
            challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);
        }
        // pushing the last challenge point to the state
        if let Some(p) = challenge {
            prover_state.challenges.push(p)
        };

        end_timer!(start);
        Ok(IOPProof {
            point: prover_state.challenges,
            proofs: prover_msgs,
        })
    }

    fn verify(
        claimed_sum: F,
        proof: &Self::SumCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "sum check verify");

        transcript.append_serializable_element(b"aux info", aux_info)?;
        let mut verifier_state = IOPVerifierState::verifier_init(aux_info);
        for i in 0..aux_info.num_variables {
            let prover_msg = proof.proofs.get(i).expect("proof is incomplete");
            transcript.append_serializable_element(b"prover msg", prover_msg)?;
            IOPVerifierState::verify_round_and_update_state(
                &mut verifier_state,
                prover_msg,
                transcript,
            )?;
        }

        let res = IOPVerifierState::check_and_generate_subclaim(&verifier_state, &claimed_sum);

        end_timer!(start);
        res
    }
}

impl<F: PrimeField> SumCheckDistributed<F> for PolyIOP<F> {
    fn prove_master(
        poly_aux_info: &Self::VPAuxInfo,
        poly_products: &Vec<(F, Vec<usize>)>,
        log_num_workers: usize,
        transcript: &mut Self::Transcript,
        master_channel: &impl MasterProverChannel,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "Distributed sum check; master");

        transcript.append_serializable_element(b"aux info", poly_aux_info)?;

        let worker_poly_aux_info = Self::VPAuxInfo {
            num_variables: poly_aux_info.num_variables - log_num_workers,
            ..poly_aux_info.clone()
        };

        let master_poly_aux_info = Self::VPAuxInfo {
            num_variables: log_num_workers,
            ..poly_aux_info.clone()
        };

        let phase1 = worker_poly_aux_info.num_variables;
        let phase2 = master_poly_aux_info.num_variables;

        if phase2 != master_channel.log_num_workers() {
            return Err(PolyIOPErrors::InvalidWorkerNumber)
        }

        if phase1 + phase2 != poly_aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidParameters(String::from("poly aux info doesn't match")))
        }

        let mut challenge = None;
        let mut challenges = Vec::new();
        let mut prover_msgs = Vec::with_capacity(phase1 + phase2);
        let eval_len = worker_poly_aux_info.max_degree + 1;

        // Phase 1:
        // 1. Master Prover sends the starting signal to the workers, workers return
        //    their virtual polynomial aux info.
        // 2. Master Prover checks the aux info matches and sends challenges to the
        //    workers sequentially, and workers return the prover messages. Master
        //    prover aggregates the worker prover messages to form the whole proof
        //    of the first phase.
        // 3. At the last round, Master Prover sends the final challenge for worker
        //    provers and worker provers send the evaluation of each mle at challenge
        //    point to the master prover.

        let phase1_timer = start_timer!(|| "phase1; master");
        master_channel.send(b"sum check starting signal")?;

        let worker_aux_infos: Vec<Self::VPAuxInfo> = master_channel.recv()?;

        worker_aux_infos.par_iter().map(|worker_aux_info|
            (worker_aux_info == &worker_poly_aux_info).then_some(()).ok_or(PolyIOPErrors::WorkerNotMatching)
        ).collect::<Result<Vec<_>, _>>()?;

        for _ in 0..phase1 {
            master_channel.send(&challenge)?;
            let worker_prover_msgs: Vec<IOPProverMessage<F>> = master_channel.recv()?;
            let evaluations = worker_prover_msgs.iter()
                .fold( vec![F::zero(); eval_len],
                    | ev1, ev2 | {
                        ev1.into_iter().zip(ev2.evaluations.iter())
                            .map(|(e1, e2)| e1 + e2)
                            .collect::<Vec<_>>()
                    }
                );

            let prover_msg = IOPProverMessage { evaluations };
            transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            prover_msgs.push(prover_msg);
            challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);

            if let Some(p) = challenge {
                challenges.push(p);
            }
        }
        end_timer!(phase1_timer);

        let construct_poly_timer = start_timer!(|| "construct poly; master");
        master_channel.send(&challenge)?;
        let flattened_ml_extensions = {
            let evals = master_channel.recv::<Vec<F>>()?;
            let len = evals.get(0).map(|x| x.len()).ok_or(PolyIOPErrors::InvalidDistributedMessage)?;
            let mut x = evals
                .into_iter().map(|x| x.into_iter())
                .collect::<Vec<_>>();
            (0..len)
                .map(
                    |_| x.iter_mut()
                    .map(|y| y.next().unwrap())
                    .collect::<Vec<_>>()
                )
                .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(phase2,x)))
                .collect::<Vec<_>>()
        };

        let poly = VirtualPolynomial::new_from_raw(
            master_poly_aux_info.clone(),
            poly_products.clone(),
            flattened_ml_extensions);

        end_timer!(construct_poly_timer);

        // Phase 2:
        //   The master prover generates the proof for the second phase
        //   just as the original sum check protocol.

        let phase2_timer = start_timer!(|| "phase2; master");
        challenge = None;

        // special situation: only one worker prover
        if phase2 == 0 {
            return Ok(IOPProof {
                point: challenges,
                proofs: prover_msgs,
            })
        }

        let mut prover_state = IOPProverState::prover_init(&poly)?;
        for _ in 0..phase2 {
            let prover_msg: IOPProverMessage<F> =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            prover_msgs.push(prover_msg);
            challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);

            if let Some(p) = challenge {
                challenges.push(p);
            }
        }

        end_timer!(phase2_timer);
        end_timer!(start);
        Ok(IOPProof {
            point: challenges,
            proofs: prover_msgs,
        })
    }

    fn prove_worker(
        poly: &Self::VirtualPolynomial,
        worker_channel: &impl WorkerProverChannel,
    ) -> Result<(), PolyIOPErrors> {
        let start = start_timer!(|| "Distributed sum check; worker");
        let start_data: [u8; 25] = worker_channel.recv()?;
        if &start_data != b"sum check starting signal" {
            return Err(PolyIOPErrors::InvalidDistributedMessage)
        }
        worker_channel.send(&poly.aux_info)?;

        let phase1_timer = start_timer!(|| "phase1; worker");
        let mut challenge = worker_channel.recv()?;

        // special situation: worker got a constant polynomial
        //    i.e., log worker number is the same as variable number
        if poly.aux_info.num_variables == 0 {
            worker_channel.send(
                &poly.flattened_ml_extensions.iter()
                    .map(|mle| mle.evaluations[0])
                    .collect::<Vec<_>>()
            )?;
            return Ok(())
        }

        let mut prover_state = IOPProverState::prover_init(poly)?;

        for i in 0..poly.aux_info.num_variables {
            let prover_msg = IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            worker_channel.send(&prover_msg)?;
            dbg!(format!("round {}, worker {}", i, worker_channel.worker_id()));
            challenge = worker_channel.recv()?;
        }

        end_timer!(phase1_timer);
        let construct_poly_timer = start_timer!(|| "construct poly; worker");

        let challenge = challenge.unwrap();
        let evaluations = prover_state.poly.flattened_ml_extensions
            .iter()
            .map(|mle| {
                let zero_value = mle.evaluations[0];
                let one_value = mle.evaluations[1];
                (one_value - zero_value) * challenge + zero_value
            })
            .collect::<Vec<_>>();

        worker_channel.send(&evaluations)?;

        end_timer!(construct_poly_timer);
        end_timer!(start);
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use crate::new_master_worker_thread_channels;

    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use std::{sync::Arc, thread::spawn};

    fn test_sumcheck(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();

        let (poly, asserted_sum) =
            VirtualPolynomial::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript)?;
        let poly_info = poly.aux_info.clone();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum,
            &proof,
            &poly_info,
            &mut transcript,
        )?;
        assert!(
            poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }

    fn test_sumcheck_internal(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let (poly, asserted_sum) =
            VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        let poly_info = poly.aux_info.clone();
        let mut prover_state = IOPProverState::prover_init(&poly)?;
        let mut verifier_state = IOPVerifierState::verifier_init(&poly_info);
        let mut challenge = None;
        let mut transcript = IOPTranscript::new(b"a test transcript");
        transcript
            .append_message(b"testing", b"initializing transcript for testing")
            .unwrap();
        for _ in 0..poly.aux_info.num_variables {
            let prover_message =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)
                    .unwrap();

            challenge = Some(
                IOPVerifierState::verify_round_and_update_state(
                    &mut verifier_state,
                    &prover_message,
                    &mut transcript,
                )
                .unwrap(),
            );
        }
        let subclaim =
            IOPVerifierState::check_and_generate_subclaim(&verifier_state, &asserted_sum)
                .expect("fail to generate subclaim");
        assert!(
            poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }

    fn test_sumcheck_distributed(
        n_log_provers: usize,
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        assert!(n_log_provers <= nv, "log number of provers should be no larger than number of variables");

        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();

        let (master_channel, worker_channels) = new_master_worker_thread_channels(n_log_provers);

        let (poly, asserted_sum) =
            VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)?;

        let distributed_poly = poly.distribute(n_log_provers)?;

        let worker_handles = worker_channels.into_iter()
            .zip(distributed_poly)
            .map( |(ch, poly)|
                (ch, poly.aux_info, poly.products, poly.flattened_ml_extensions)
            )
            .map( |(ch, aux_info, products, mles)| {
                spawn(move || {
                    <PolyIOP<Fr> as SumCheckDistributed<Fr>>::prove_worker(
                        &VirtualPolynomial::new_from_raw(aux_info, products, mles), &ch)
                })
            }).collect::<Vec<_>>();

        let proof = <PolyIOP<Fr> as SumCheckDistributed<Fr>>::prove_master(
            &poly.aux_info, &poly.products, n_log_provers, &mut transcript, &master_channel)?;

        worker_handles.into_iter().map(|x| x.join().unwrap())
            .collect::<Result<Vec<_>, _>>()?;

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum,
            &proof,
            &poly.aux_info,
            &mut transcript,
        )?;
        assert!(
            poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );

        Ok(())
    }

    #[test]
    fn test_distributed_polynomial() -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let n_log_provers = 2;
        let nv = 8;
        let num_multiplicands_range = (1, 3);
        let num_products = 5;

        let (poly, _) = VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        let distributed_poly = poly.distribute(n_log_provers)?;

        assert_eq!(
            poly.evaluate(&vec![Fr::from(0); nv])?,
            distributed_poly[0].evaluate(&vec![Fr::from(0); nv - n_log_provers])?,
            "wrong evaluation"
        );

        test_sumcheck_distributed(n_log_provers, nv, num_multiplicands_range, num_products)?;
        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 1;
        let num_multiplicands_range = (4, 13);
        let num_products = 5;

        test_sumcheck(nv, num_multiplicands_range, num_products)?;
        test_sumcheck_internal(nv, num_multiplicands_range, num_products)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 12;
        let num_multiplicands_range = (4, 9);
        let num_products = 5;

        test_sumcheck(nv, num_multiplicands_range, num_products)?;
        test_sumcheck_internal(nv, num_multiplicands_range, num_products)
    }
    #[test]
    fn zero_polynomial_should_error() {
        let nv = 0;
        let num_multiplicands_range = (4, 13);
        let num_products = 5;

        assert!(test_sumcheck(nv, num_multiplicands_range, num_products).is_err());
        assert!(test_sumcheck_internal(nv, num_multiplicands_range, num_products).is_err());
    }

    #[test]
    fn test_extract_sum() -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (poly, asserted_sum) = VirtualPolynomial::<Fr>::rand(8, (3, 4), 3, &mut rng)?;

        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript)?;
        assert_eq!(
            <PolyIOP<Fr> as SumCheck<Fr>>::extract_sum(&proof),
            asserted_sum
        );
        Ok(())
    }

    #[test]
    /// Test that the memory usage of shared-reference is linear to number of
    /// unique MLExtensions instead of total number of multiplicands.
    fn test_shared_reference() -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let ml_extensions: Vec<_> = (0..5)
            .map(|_| Arc::new(DenseMultilinearExtension::<Fr>::rand(8, &mut rng)))
            .collect();
        let mut poly = VirtualPolynomial::new(8);
        poly.add_mle_list(
            vec![
                ml_extensions[2].clone(),
                ml_extensions[3].clone(),
                ml_extensions[0].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![
                ml_extensions[1].clone(),
                ml_extensions[4].clone(),
                ml_extensions[4].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![
                ml_extensions[3].clone(),
                ml_extensions[2].clone(),
                ml_extensions[1].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![ml_extensions[0].clone(), ml_extensions[0].clone()],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(vec![ml_extensions[4].clone()], Fr::rand(&mut rng))?;

        assert_eq!(poly.flattened_ml_extensions.len(), 5);

        // test memory usage for prover
        let prover = IOPProverState::<Fr>::prover_init(&poly).unwrap();
        assert_eq!(prover.poly.flattened_ml_extensions.len(), 5);
        drop(prover);

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let poly_info = poly.aux_info.clone();
        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript)?;
        let asserted_sum = <PolyIOP<Fr> as SumCheck<Fr>>::extract_sum(&proof);

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum,
            &proof,
            &poly_info,
            &mut transcript,
        )?;
        assert!(
            poly.evaluate(&subclaim.point)? == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }
}
