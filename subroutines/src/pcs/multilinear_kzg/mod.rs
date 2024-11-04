// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for multilinear KZG commitment scheme

pub(crate) mod batching;
pub(crate) mod srs;
pub(crate) mod util;

use crate::{
    pcs::{prelude::Commitment, PCSError, PolynomialCommitmentScheme, StructuredReferenceString},
    BatchProof, MasterProverChannel, PolyIOP, SumCheckDistributed, WorkerProverChannel,
};
use arithmetic::{
    build_eq_x_r, build_eq_x_r_vec, eq_eval, evaluate_opt, start_timer_with_timestamp, transpose,
    VPAuxInfo, VirtualPolynomial,
};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    AffineRepr, CurveGroup,
};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, end_timer, format, log2, marker::PhantomData, rand::Rng, string::ToString,
    sync::Arc, vec::Vec, One, Zero,
};
use std::ops::Mul;
// use batching::{batch_verify_internal, multi_open_internal};
use srs::{
    Evaluations, MultilinearProverParam, MultilinearUniversalParams, MultilinearVerifierParam,
};
use transcript::IOPTranscript;

use self::batching::{batch_verify_internal, multi_open_internal};

use super::PolynomialCommitmentSchemeDistributed;

/// KZG Polynomial Commitment Scheme on multilinear polynomials.
pub struct MultilinearKzgPCS<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
/// proof of opening
pub struct MultilinearKzgProof<E: Pairing> {
    /// Evaluation of quotients
    pub proofs: Vec<E::G1Affine>,
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for MultilinearKzgPCS<E> {
    // Parameters
    type ProverParam = MultilinearProverParam<E>;
    type VerifierParam = MultilinearVerifierParam<E>;
    type SRS = MultilinearUniversalParams<E>;
    // Polynomial and its associated types
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // Commitments and proofs
    type Commitment = Commitment<E>;
    type Proof = MultilinearKzgProof<E>;
    type BatchProof = BatchProof<E, Self>;

    /// Build SRS for testing.
    ///
    /// - For univariate polynomials, `log_size` is the log of maximum degree.
    /// - For multilinear polynomials, `log_size` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(rng: &mut R, log_size: usize) -> Result<Self::SRS, PCSError> {
        MultilinearUniversalParams::<E>::gen_srs_for_testing(rng, log_size)
    }

    /// Trim the universal parameters to specialize the public parameters.
    /// Input both `supported_log_degree` for univariate and
    /// `supported_num_vars` for multilinear.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        assert!(supported_degree.is_none());

        let supported_num_vars = match supported_num_vars {
            Some(p) => p,
            None => {
                return Err(PCSError::InvalidParameters(
                    "multilinear should receive a num_var param".to_string(),
                ))
            },
        };
        let (ml_ck, ml_vk) = srs.borrow().trim(supported_num_vars)?;

        Ok((ml_ck, ml_vk))
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    fn commit(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let prover_param = prover_param.borrow();
        let commit_timer = start_timer_with_timestamp!("commit");
        if prover_param.num_vars < poly.num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "MlE length ({}) exceeds param limit ({})",
                poly.num_vars, prover_param.num_vars
            )));
        }
        let ignored = prover_param.num_vars - poly.num_vars;
        let scalars: Vec<_> = poly.to_evaluations();
        let msm_timer = start_timer_with_timestamp!(format!(
            "msm of size {}",
            prover_param.powers_of_g[ignored].evals.len()
        ));
        let commitment =
            E::G1::msm_unchecked(&prover_param.powers_of_g[ignored].evals, scalars.as_slice())
                .into_affine();
        end_timer!(msm_timer);

        end_timer!(commit_timer);
        Ok(Commitment(commitment))
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the
    /// same. This function does not need to take the evaluation value as an
    /// input.
    ///
    /// This function takes 2^{num_var +1} number of scalar multiplications over
    /// G1:
    /// - it prodceeds with `num_var` number of rounds,
    /// - at round i, we compute an MSM for `2^{num_var - i + 1}` number of G2
    ///   elements.
    fn open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomial: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        open_internal(prover_param.borrow(), polynomial, point)
    }

    /// Input a list of multilinear extensions, and a same number of points, and
    /// a transcript, compute a multi-opening for all the polynomials.
    fn multi_open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<BatchProof<E, Self>, PCSError> {
        multi_open_internal(
            prover_param.borrow(),
            polynomials,
            points,
            evals,
            transcript,
        )
    }

    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    ///
    /// This function takes
    /// - num_var number of pairing product.
    /// - num_var number of MSM
    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        verify_internal(verifier_param, commitment, point, value, proof)
    }

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        batch_verify_internal(verifier_param, commitments, points, batch_proof, transcript)
    }
}

impl<E: Pairing> PolynomialCommitmentSchemeDistributed<E> for MultilinearKzgPCS<E> {
    type MasterProverParam = MultilinearProverParam<E>;
    type WorkerProverParam = MultilinearProverParam<E>;

    type MasterPolynomialHandle = usize;
    type WorkerPolynomialHandle = Self::Polynomial;

    /// Distribute the prover parameter generated by
    /// PolynomialCommitmentScheme::trim into master and worker parameters,
    /// both of which are of the same type.
    ///
    /// Originally, the prover parameter contains a list of powers of g, which
    /// can be visually represented as follows:
    ///       powers_of_g = [[0:2^num_vars], [0:2^(num_vars-1)], ..., [0:2^0]]
    ///
    /// The master parameter contains the last `log_num_workers` number of
    /// powers of g, while the worker parameters distribute evenly the
    /// remaining powers of g, with each taking its share of every element
    /// in powers_of_g.
    fn prover_param_distributed(
        prover_param: Self::ProverParam,
        log_num_workers: usize,
    ) -> Result<(Self::MasterProverParam, Vec<Self::WorkerProverParam>), PCSError> {
        let num_workers = 1 << log_num_workers;
        if prover_param.num_vars < log_num_workers {
            return Err(PCSError::InvalidParameters(format!(
                "num_vars {} < log_num_workers {}",
                prover_param.num_vars, log_num_workers
            )));
        }

        let total_num_vars = prover_param.num_vars;
        let worker_num_vars = total_num_vars - log_num_workers;
        let master_powers_of_g = prover_param.powers_of_g[worker_num_vars..].to_vec();
        let (g, h) = (prover_param.g, prover_param.h);
        let worker_params = {
            let powers_of_g = prover_param.powers_of_g;
            let mut iters = powers_of_g
                .into_iter()
                .take(total_num_vars - log_num_workers + 1)
                .map(|x| x.evals.into_iter())
                .collect::<Vec<_>>();

            (0..num_workers)
                .map(|_| {
                    let powers_of_g = (0..worker_num_vars + 1)
                        .rev()
                        .map(|i| 1 << i)
                        .zip(iters.iter_mut())
                        .map(|(size, it)| Evaluations {
                            evals: it.take(size).collect::<Vec<_>>(),
                        })
                        .collect::<Vec<_>>();
                    Self::WorkerProverParam {
                        num_vars: worker_num_vars,
                        powers_of_g,
                        g,
                        h,
                    }
                })
                .collect::<Vec<_>>()
        };

        let master_param = Self::MasterProverParam {
            num_vars: log_num_workers,
            powers_of_g: master_powers_of_g,
            g,
            h,
        };

        Ok((master_param, worker_params))
    }

    /// Commit to a polynomial in a distributed manner.
    ///
    /// Arguments:
    /// - `master_prover_param`: the master prover parameter returned by
    ///   `prover_param_distributed`.
    /// - `handle`: the handle of the multilinear extension to commit to, which
    ///   is the number of
    /// variables of the multilinear extension.
    /// - `master_channel`: the channel to communicate with the workers.
    ///
    /// Returns:
    /// - the commitment to the polynomial.
    fn commit_distributed_master(
        master_prover_param: impl Borrow<Self::MasterProverParam>,
        _handle: &Self::MasterPolynomialHandle,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<Self::Commitment, PCSError> {
        let timer = start_timer_with_timestamp!("multilinear KZG: commit; master");
        let master_num_vars = master_prover_param.borrow().num_vars;

        if master_num_vars != master_channel.log_num_workers() {
            return Err(PCSError::InvalidParameters(format!(
                "master_num_vars {} != log_num_workers {}",
                master_num_vars,
                master_channel.log_num_workers()
            )));
        }

        let commitments: Vec<E::G1Affine> = master_channel.recv()?;
        // commitments.iter().fold(E::G1Affine::from(1), |acc, x| acc * x.0);

        let msm_timer =
            start_timer_with_timestamp!(format!("msm of size {}; master", commitments.len()));
        let commitment = E::G1::msm_unchecked(
            &commitments,
            &vec![<E as Pairing>::ScalarField::from(1u128); 1 << master_num_vars],
        )
        .into_affine();

        end_timer!(msm_timer);
        end_timer!(timer);
        Ok(Commitment(commitment))
    }

    /// Commit to a polynomial in a distributed manner.
    ///
    /// Arguments:
    /// - `worker_prover_param`: the worker prover parameter returned by
    ///   `prover_param_distributed`.
    /// - `poly`: the worker's share of the multilinear extension to commit to.
    /// - `worker_channel`: the channel to communicate with the master.
    fn commit_distributed_worker(
        worker_prover_param: impl Borrow<Self::WorkerProverParam>,
        poly: &Self::WorkerPolynomialHandle,
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(), PCSError> {
        let worker_prover_param = worker_prover_param.borrow();
        let commit_timer = start_timer_with_timestamp!(format!(
            "multilinear KZG: commit; worker_id {}",
            worker_channel.worker_id()
        ));

        if worker_prover_param.num_vars != poly.num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "MlE length ({}) not equal to prover params ({})",
                poly.num_vars, worker_prover_param.num_vars
            )));
        }

        let scalars: Vec<_> = poly.to_evaluations();
        let msm_timer = start_timer_with_timestamp!(format!(
            "msm of size {}; worker_id {}",
            worker_prover_param.powers_of_g[0].evals.len(),
            worker_channel.worker_id()
        ));
        let commitment: E::G1Affine = E::G1::msm_unchecked(
            &worker_prover_param.powers_of_g[0].evals,
            scalars.as_slice(),
        )
        .into_affine();
        end_timer!(msm_timer);

        worker_channel.send(&commitment)?;
        end_timer!(commit_timer);
        Ok(())
    }

    /// Open a polynomial in a distributed manner.
    ///
    /// The master prover
    ///
    /// Arguments:
    /// - `master_prover_param`: the master prover parameter returned by
    ///   `prover_param_distributed`.
    /// - `handle`: the handle of the multilinear extension to open, which is
    ///   the number of variables
    /// of the multilinear extension.
    /// - `point`: the point at which to open the multilinear extension. The
    ///   length of the point
    /// should be equal to the handle.
    /// - `master_channel`: the channel to communicate with the workers.
    ///
    /// Returns:
    /// - the proof of the opening and the evaluation of the polynomial at the
    ///   point.
    fn open_distributed_master(
        master_prover_param: impl Borrow<Self::MasterProverParam>,
        handle: &Self::MasterPolynomialHandle,
        point: &Self::Point,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let timer = start_timer_with_timestamp!("multilinear KZG: open; master");

        let master_num_vars = master_prover_param.borrow().num_vars;
        let worker_num_vars = *handle - master_num_vars;

        if master_num_vars != master_channel.log_num_workers() {
            return Err(PCSError::InvalidParameters(format!(
                "master_num_vars {} != log_num_workers {}",
                master_num_vars,
                master_channel.log_num_workers()
            )));
        }

        if point.len() != *handle {
            return Err(PCSError::InvalidParameters(format!(
                "point length ({}) not equal to handle ({})",
                point.len(),
                handle
            )));
        }

        if master_num_vars > point.len() {
            return Err(PCSError::InvalidParameters(format!(
                "master_num_vars {} > point length {}",
                master_num_vars,
                point.len()
            )));
        }

        let (worker_points, master_points) = point.split_at(worker_num_vars);

        master_channel.send_uniform(b"open starting signal")?;
        master_channel.send_uniform(&worker_points.to_vec())?;
        let evals: Vec<Self::Evaluation> = master_channel.recv()?;
        let master_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            master_num_vars,
            evals,
        ));

        let (proof, eval) =
            open_internal(master_prover_param.borrow(), &master_poly, master_points)?;

        let worker_proofs: Vec<MultilinearKzgProof<E>> = master_channel.recv()?;

        let aggregated_proof = {
            let mut proofs_iter = worker_proofs
                .into_iter()
                .map(|x| x.proofs.into_iter())
                .collect::<Vec<_>>();
            let mut x = (0..worker_num_vars)
                .map(|_| {
                    let acc = proofs_iter
                        .iter_mut()
                        .map(|x| x.next().unwrap())
                        .collect::<Vec<_>>();
                    E::G1::msm_unchecked(
                        &acc,
                        &vec![<E as Pairing>::ScalarField::from(1u128); 1 << master_num_vars],
                    )
                    .into_affine()
                })
                .collect::<Vec<_>>();
            x.extend(proof.proofs.iter());
            x
        };

        end_timer!(timer);

        Ok((
            MultilinearKzgProof {
                proofs: aggregated_proof,
            },
            eval,
        ))
    }

    /// Open a polynomial in a distributed manner.
    ///
    /// The worker prover only has its share of the multilinear extension,
    /// waiting for the master prover to send the opening point to open the
    /// multilinear extension.
    ///
    /// Arguments:
    /// - `worker_prover_param`: the worker prover parameter returned by
    ///   `prover_param_distributed`.
    /// - `poly`: the worker's share of the multilinear extension to open.
    /// - `worker_channel`: the channel to communicate with the master.
    fn open_distributed_worker(
        worker_prover_param: impl Borrow<Self::WorkerProverParam>,
        poly: &Self::WorkerPolynomialHandle,
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(), PCSError> {
        let timer = start_timer_with_timestamp!(format!(
            "multilinear KZG: open; worker_id {}",
            worker_channel.worker_id()
        ));

        if worker_prover_param.borrow().num_vars != poly.num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "MlE length ({}) not equal to prover params ({})",
                poly.num_vars,
                worker_prover_param.borrow().num_vars
            )));
        }

        let start_msg: [u8; 20] = worker_channel.recv()?;
        if &start_msg != b"open starting signal" {
            return Err(PCSError::InvalidParameters(format!(
                "Received unexpected message: {:?}",
                start_msg
            )));
        }

        let point: Self::Point = worker_channel.recv()?;
        worker_channel.send(&poly.evaluate(&point).unwrap())?;

        let (proof, _) = open_internal(worker_prover_param.borrow(), poly, &point)?;
        worker_channel.send(&proof)?;

        end_timer!(timer);
        Ok(())
    }

    /// Generate a batch proof for multiple openings of polynomials at different
    /// points in a distributed manner.
    ///
    /// Arguments:
    /// - `master_prover_param`: the master prover parameter returned by
    ///   `prover_param_distributed`.
    /// - `handle`: the handle of the multilinear extension to open, which is
    ///   the number of variables
    /// of the multilinear extension.
    /// - `points`: the points at which to open the multilinear extensions. The
    ///   length of each point
    /// should be equal to the handle.
    /// - `transcript`: the transcript to use for the proof.
    /// - `master_channel`: the channel to communicate with the workers.
    fn multi_open_master(
        master_prover_param: impl Borrow<Self::MasterProverParam>,
        handle: &Self::MasterPolynomialHandle,
        points: &[Self::Point],
        transcript: &mut IOPTranscript<E::ScalarField>,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<Self::BatchProof, PCSError> {
        let timer = start_timer_with_timestamp!("multilinear KZG: multiple open; master");

        let num_var = *handle;
        let log_num_workers = master_channel.log_num_workers();
        let worker_num_vars = num_var - log_num_workers;
        let k = points.len();
        let ell = log2(k) as usize;

        // send the challenge point t to the workers
        let t: Vec<E::ScalarField> =
            transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
        master_channel.send_uniform(&t)?;

        let mut worker_points = Vec::new();
        let mut master_points = Vec::new();
        for point in points.iter() {
            let (worker_point, master_point) = point.split_at(worker_num_vars);
            worker_points.push(worker_point.to_vec());
            master_points.push(master_point.to_vec());
        }

        master_channel.send_uniform(&worker_points)?;

        master_channel.send_different(transpose(
            master_points
                .iter()
                .map(|point| build_eq_x_r_vec(point.as_ref()))
                .collect::<Result<Vec<_>, _>>()?,
        ))?;

        let evals: Vec<Vec<Self::Evaluation>> = master_channel.recv()?;
        let evals = transpose(evals)
            .into_iter()
            .zip(master_points.iter())
            .map(|(evals, point)| {
                evaluate_opt(
                    &DenseMultilinearExtension::from_evaluations_vec(log_num_workers, evals),
                    point,
                )
            })
            .collect::<Vec<_>>();

        // generate the sumcheck proof
        let proof =
            match <PolyIOP<E::ScalarField> as SumCheckDistributed<E::ScalarField>>::prove_master(
                &VPAuxInfo {
                    max_degree: 2,
                    num_variables: num_var,
                    phantom: PhantomData,
                },
                // [(1, [0, k]), (1, [1, k+1]), ..., (1, [k-1, 2k-1])]
                &(0..k)
                    .map(|i| (E::ScalarField::one(), vec![i, i + k]))
                    .collect::<Vec<_>>(),
                log_num_workers,
                transcript,
                master_channel,
            ) {
                Ok(proof) => proof,
                Err(_e) => {
                    return Err(PCSError::InvalidProver(
                        "SumCheckDistributed on master failed".to_string(),
                    ))
                },
            };

        let challenge = &proof.point;
        let tilde_eqs_master_scalar = points
            .iter()
            .map(|point| eq_eval(point, challenge))
            .collect::<Result<Vec<_>, _>>()?;
        master_channel.send_uniform(&tilde_eqs_master_scalar)?;

        // generate the opening proof for g_prime
        let (g_prime_proof, _) = Self::open_distributed_master(
            master_prover_param.borrow(),
            handle,
            challenge,
            master_channel,
        )?;

        end_timer!(timer);
        Ok(BatchProof {
            sum_check_proof: proof,
            f_i_eval_at_point_i: evals,
            g_prime_proof,
        })
    }

    /// Generate a batch proof for multiple openings of polynomials at different
    /// points in a distributed manner.
    ///
    /// Arguments:
    /// - `worker_prover_param`: the worker prover parameter returned by
    ///   `prover_param_distributed`.
    /// - `polynomials`: the worker's shares of the multilinear extensions to
    ///   open.
    /// - `worker_channel`: the channel to communicate with the master.
    fn multi_open_worker(
        worker_prover_param: impl Borrow<Self::WorkerProverParam>,
        polynomials: &[Self::WorkerPolynomialHandle],
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(), PCSError> {
        let timer = start_timer_with_timestamp!(format!(
            "multilinear KZG: multiple open; worker_id {}",
            worker_channel.worker_id()
        ));

        let k = polynomials.len();
        let num_vars = polynomials[0].num_vars;
        let ell = log2(k) as usize;

        let t: Vec<E::ScalarField> = worker_channel.recv()?;
        assert_eq!(t.len(), ell);
        let eq_t_i_list = build_eq_x_r(t.as_ref())?;

        let points: Vec<Self::Point> = worker_channel.recv()?;
        assert_eq!(points.len(), k);
        let evals = points
            .iter()
            .zip(polynomials.iter())
            .map(|(point, poly)| evaluate_opt(poly, point))
            .collect::<Vec<_>>();
        worker_channel.send(&evals)?;

        let tilde_eqs_master_scalar: Vec<Self::Evaluation> = worker_channel.recv()?;
        assert_eq!(tilde_eqs_master_scalar.len(), k);
        let tilde_eqs = tilde_eqs_master_scalar
            .iter()
            .zip(points.iter())
            .map(|(scalar, point)| -> Result<Self::Polynomial, PCSError> {
                Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    build_eq_x_r_vec(point.as_ref())?
                        .iter()
                        .map(|eq| *eq * scalar)
                        .collect::<Vec<_>>(),
                )))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let tilde_gs = polynomials
            .iter()
            .zip(eq_t_i_list.iter())
            .take(k)
            .map(|(poly, eq_t_i)| {
                Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                    num_vars,
                    &*poly
                        .evaluations
                        .iter()
                        .map(|eval| *eval * eq_t_i)
                        .collect::<Vec<_>>(),
                ))
            })
            .collect::<Vec<_>>();

        let sum_check_vp = VirtualPolynomial::new_from_raw(
            VPAuxInfo {
                max_degree: 2,
                num_variables: num_vars,
                phantom: PhantomData,
            },
            // [(1, [0, k]), (1, [1, k+1]), ..., (1, [k-1, 2k-1])]
            (0..k)
                .map(|i| (E::ScalarField::one(), vec![i, i + k]))
                .collect::<Vec<_>>(),
            tilde_gs
                .iter()
                .chain(tilde_eqs.iter())
                .cloned()
                .collect::<Vec<_>>(),
        );

        match <PolyIOP<E::ScalarField> as SumCheckDistributed<E::ScalarField>>::prove_worker(
            &sum_check_vp,
            worker_channel,
        ) {
            Ok(_) => (),
            Err(_e) => {
                return Err(PCSError::InvalidProver(
                    "SumCheckDistributed on worker failed".to_string(),
                ))
            },
        };

        let tilde_eqs_eval_scalar: Vec<Self::Evaluation> = worker_channel.recv()?;
        let mut g_prime = Arc::new(DenseMultilinearExtension::zero());
        for (tilde_g, tilde_eq_eval) in tilde_gs.iter().zip(tilde_eqs_eval_scalar.iter()) {
            *Arc::make_mut(&mut g_prime) += (*tilde_eq_eval, &**tilde_g);
        }

        Self::open_distributed_worker(worker_prover_param.borrow(), &g_prime, worker_channel)?;

        end_timer!(timer);
        Ok(())
    }
}

/// On input a polynomial `p` and a point `point`, outputs a proof for the
/// same. This function does not need to take the evaluation value as an
/// input.
///
/// This function takes 2^{num_var} number of scalar multiplications over
/// G1:
/// - it proceeds with `num_var` number of rounds,
/// - at round i, we compute an MSM for `2^{num_var - i}` number of G1 elements.
fn open_internal<E: Pairing>(
    prover_param: &MultilinearProverParam<E>,
    polynomial: &DenseMultilinearExtension<E::ScalarField>,
    point: &[E::ScalarField],
) -> Result<(MultilinearKzgProof<E>, E::ScalarField), PCSError> {
    let open_timer =
        start_timer_with_timestamp!(format!("open mle with {} variable", polynomial.num_vars));

    if polynomial.num_vars() > prover_param.num_vars {
        return Err(PCSError::InvalidParameters(format!(
            "Polynomial num_vars {} exceed the limit {}",
            polynomial.num_vars, prover_param.num_vars
        )));
    }

    if polynomial.num_vars() != point.len() {
        return Err(PCSError::InvalidParameters(format!(
            "Polynomial num_vars {} does not match point len {}",
            polynomial.num_vars,
            point.len()
        )));
    }

    let nv = polynomial.num_vars();
    // the first `ignored` SRS vectors are unused for opening.
    let ignored = prover_param.num_vars - nv + 1;
    let mut f = polynomial.to_evaluations();

    let mut proofs = Vec::new();

    for (i, (&point_at_k, gi)) in point
        .iter()
        .zip(prover_param.powers_of_g[ignored..ignored + nv].iter())
        .enumerate()
    {
        let ith_round = start_timer_with_timestamp!(format!("{}-th round", i));

        let k = nv - 1 - i;
        let cur_dim = 1 << k;
        let mut q = vec![E::ScalarField::zero(); cur_dim];
        let mut r = vec![E::ScalarField::zero(); cur_dim];

        let ith_round_eval = start_timer_with_timestamp!(format!("{}-th round eval", i));
        for b in 0..(1 << k) {
            // q[b] = f[1, b] - f[0, b]
            q[b] = f[(b << 1) + 1] - f[b << 1];

            // r[b] = f[0, b] + q[b] * p
            r[b] = f[b << 1] + (q[b] * point_at_k);
        }
        f = r;
        end_timer!(ith_round_eval);

        // this is a MSM over G1 and is likely to be the bottleneck
        let msm_timer = start_timer_with_timestamp!(format!(
            "msm of size {} at round {}",
            gi.evals.len(),
            i
        ));

        proofs.push(E::G1::msm_unchecked(&gi.evals, &q).into_affine());
        end_timer!(msm_timer);

        end_timer!(ith_round);
    }
    let eval = evaluate_opt(polynomial, point);
    end_timer!(open_timer);
    Ok((MultilinearKzgProof { proofs }, eval))
}

/// Verifies that `value` is the evaluation at `x` of the polynomial
/// committed inside `comm`.
///
/// This function takes
/// - num_var number of pairing product.
/// - num_var number of MSM
fn verify_internal<E: Pairing>(
    verifier_param: &MultilinearVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MultilinearKzgProof<E>,
) -> Result<bool, PCSError> {
    let verify_timer = start_timer_with_timestamp!("verify");
    let num_var = point.len();

    if num_var > verifier_param.num_vars {
        return Err(PCSError::InvalidParameters(format!(
            "point length ({}) exceeds param limit ({})",
            num_var, verifier_param.num_vars
        )));
    }

    let prepare_inputs_timer = start_timer_with_timestamp!("prepare pairing inputs");

    let scalar_size = E::ScalarField::MODULUS_BIT_SIZE as usize;
    let window_size = FixedBase::get_mul_window_size(num_var);

    let h_table =
        FixedBase::get_window_table(scalar_size, window_size, verifier_param.h.into_group());
    let h_mul: Vec<E::G2> = FixedBase::msm(scalar_size, window_size, &h_table, point);

    let ignored = verifier_param.num_vars - num_var;
    let h_vec: Vec<_> = (0..num_var)
        .map(|i| verifier_param.h_mask[ignored + i].into_group() - h_mul[i])
        .collect();
    let h_vec: Vec<E::G2Affine> = E::G2::normalize_batch(&h_vec);
    end_timer!(prepare_inputs_timer);

    let pairing_product_timer = start_timer_with_timestamp!("pairing product");

    let mut pairings: Vec<_> = proof
        .proofs
        .iter()
        .map(|&x| E::G1Prepared::from(x))
        .zip(h_vec.into_iter().take(num_var).map(E::G2Prepared::from))
        .collect();

    pairings.push((
        E::G1Prepared::from(
            (verifier_param.g.mul(*value) - commitment.0.into_group()).into_affine(),
        ),
        E::G2Prepared::from(verifier_param.h),
    ));

    let ps = pairings.iter().map(|(p, _)| p.clone());
    let hs = pairings.iter().map(|(_, h)| h.clone());

    let res = E::multi_pairing(ps, hs) == ark_ec::pairing::PairingOutput(E::TargetField::one());

    end_timer!(pairing_product_timer);
    end_timer!(verify_timer);
    Ok(res)
}

#[cfg(test)]
mod tests {
    use std::thread::spawn;

    use crate::new_master_worker_channels;

    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{test_rng, vec::Vec, UniformRand};

    type E = Bls12_381;
    type Fr = <E as Pairing>::ScalarField;

    fn test_single_helper<R: Rng>(
        params: &MultilinearUniversalParams<E>,
        poly: &Arc<DenseMultilinearExtension<Fr>>,
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let nv = poly.num_vars();
        assert_ne!(nv, 0);
        let (ck, vk) = MultilinearKzgPCS::trim(params, None, Some(nv))?;
        let point: Vec<_> = (0..nv).map(|_| Fr::rand(rng)).collect();
        let com = MultilinearKzgPCS::commit(&ck, poly)?;
        let (proof, value) = MultilinearKzgPCS::open(&ck, poly, &point)?;

        assert!(MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        let value = Fr::rand(rng);
        assert!(!MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        Ok(())
    }

    fn test_single_helper_distributed<R: Rng>(
        params: &MultilinearUniversalParams<E>,
        polys: Vec<Arc<DenseMultilinearExtension<Fr>>>,
        log_num_workers: usize,
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let nv = polys[0].num_vars() + log_num_workers;
        assert_eq!(polys.len(), 1 << log_num_workers);
        let (ck, vk) = MultilinearKzgPCS::trim(params, None, Some(nv))?;

        let (master_ck, worker_ck) =
            MultilinearKzgPCS::prover_param_distributed(ck, log_num_workers)?;

        let (mut master_channel, worker_channel) =
            new_master_worker_channels(true, log_num_workers, "127.0.0.1:0");

        // let (mut master_channel, worker_channel) =
        // new_master_worker_thread_channels(log_num_workers);

        let handles: Vec<_> = worker_ck
            .into_iter()
            .zip(polys)
            .zip(worker_channel)
            .map(|((ck, poly), mut ch)| {
                spawn(move || {
                    MultilinearKzgPCS::commit_distributed_worker(&ck, &poly, &mut ch)?;
                    MultilinearKzgPCS::open_distributed_worker(&ck, &poly, &mut ch)
                })
            })
            .collect();

        let com =
            MultilinearKzgPCS::commit_distributed_master(&master_ck, &nv, &mut master_channel)?;
        let point: Vec<_> = (0..nv).map(|_| Fr::rand(rng)).collect();
        let (proof, value) = MultilinearKzgPCS::open_distributed_master(
            &master_ck,
            &nv,
            &point,
            &mut master_channel,
        )?;

        handles
            .into_iter()
            .map(|x| x.join().unwrap())
            .collect::<Result<Vec<_>, PCSError>>()?;

        assert!(MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        let value = Fr::rand(rng);
        assert!(!MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        let mut proof = proof;
        proof.proofs[0] = <E as Pairing>::G1Affine::zero();
        assert!(!MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        Ok(())
    }

    #[test]
    fn test_single_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

        // normal polynomials
        let poly1 = Arc::new(DenseMultilinearExtension::rand(8, &mut rng));
        test_single_helper(&params, &poly1, &mut rng)?;

        // single-variate polynomials
        let poly2 = Arc::new(DenseMultilinearExtension::rand(1, &mut rng));
        test_single_helper(&params, &poly2, &mut rng)?;

        Ok(())
    }

    #[test]
    fn test_single_commit_distributed() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let log_num_workers = 4;
        let worker_num_vars = 5;

        let polys = (0..(1 << log_num_workers))
            .map(|_| Arc::new(DenseMultilinearExtension::rand(worker_num_vars, &mut rng)))
            .collect::<Vec<_>>();

        test_single_helper_distributed(&params, polys, log_num_workers, &mut rng)?;
        Ok(())
    }

    #[test]
    fn setup_commit_verify_constant_polynomial() {
        let mut rng = test_rng();

        // normal polynomials
        assert!(MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 0).is_err());
    }
}
