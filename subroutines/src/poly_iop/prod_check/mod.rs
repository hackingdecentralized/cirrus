// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the Product Check protocol

use crate::{
    pcs::{PolynomialCommitmentScheme, PolynomialCommitmentSchemeDistributed},
    poly_iop::{
        errors::PolyIOPErrors,
        prod_check::util::{compute_frac_poly, compute_product_poly, prove_zero_check},
        zero_check::{ZeroCheck, ZeroCheckDistributed},
        PolyIOP,
    },
    MasterProverChannel, MultilinearProverParam, WorkerProverChannel,
};
use arithmetic::{get_index, VPAuxInfo, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_ff::{One, PrimeField, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use std::{fmt::Debug, sync::Arc};
use transcript::IOPTranscript;

mod util;

/// A product-check proves that two lists of n-variate multilinear polynomials
/// `(f1, f2, ..., fk)` and `(g1, ..., gk)` satisfy:
/// \prod_{x \in {0,1}^n} f1(x) * ... * fk(x) = \prod_{x \in {0,1}^n} g1(x) *
/// ... * gk(x)
///
/// A ProductCheck is derived from ZeroCheck.
///
/// Prover steps:
/// 1. build MLE `frac(x)` s.t. `frac(x) = f1(x) * ... * fk(x) / (g1(x) * ... *
/// gk(x))` for all x \in {0,1}^n 2. build `prod(x)` from `frac(x)`, where
/// `prod(x)` equals to `v(1,x)` in the paper 2. push commitments of `frac(x)`
/// and `prod(x)` to the transcript,    and `generate_challenge` from current
/// transcript (generate alpha) 3. generate the zerocheck proof for the virtual
/// polynomial Q(x):       prod(x) - p1(x) * p2(x)
///     + alpha * frac(x) * g1(x) * ... * gk(x)
///     - alpha * f1(x) * ... * fk(x)
/// where p1(x) = (1-x1) * frac(x2, ..., xn, 0)
///             + x1 * prod(x2, ..., xn, 0),
/// and p2(x) = (1-x1) * frac(x2, ..., xn, 1)
///           + x1 * prod(x2, ..., xn, 1)
///
/// Verifier steps:
/// 1. Extract commitments of `frac(x)` and `prod(x)` from the proof, push
/// them to the transcript
/// 2. `generate_challenge` from current transcript (generate alpha)
/// 3. `verify` to verify the zerocheck proof and generate the subclaim for
/// polynomial evaluations
pub trait ProductCheck<E, PCS>: ZeroCheck<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type ProductCheckSubClaim;
    type ProductCheckProof;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a ProductCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// ProductCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// Proves that two lists of n-variate multilinear polynomials `(f1, f2,
    /// ..., fk)` and `(g1, ..., gk)` satisfy:
    ///   \prod_{x \in {0,1}^n} f1(x) * ... * fk(x)
    /// = \prod_{x \in {0,1}^n} g1(x) * ... * gk(x)
    ///
    /// Inputs:
    /// - fxs: the list of numerator multilinear polynomial
    /// - gxs: the list of denominator multilinear polynomial
    /// - transcript: the IOP transcript
    /// - pk: PCS committing key
    ///
    /// Outputs
    /// - the product check proof
    /// - the product polynomial (used for testing)
    /// - the fractional polynomial (used for testing)
    ///
    /// Cost: O(N)
    #[allow(clippy::type_complexity)]
    fn prove(
        pcs_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::ProductCheckProof,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
        ),
        PolyIOPErrors,
    >;

    /// Verify that for witness multilinear polynomials (f1, ..., fk, g1, ...,
    /// gk) it holds that
    ///      `\prod_{x \in {0,1}^n} f1(x) * ... * fk(x)
    ///     = \prod_{x \in {0,1}^n} g1(x) * ... * gk(x)`
    fn verify(
        proof: &Self::ProductCheckProof,
        aux_info: &VPAuxInfo<E::ScalarField>,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ProductCheckSubClaim, PolyIOPErrors>;
}

/// A product check subclaim consists of
/// - A zero check IOP subclaim for the virtual polynomial
/// - The random challenge `alpha`
/// - A final query for `prod(1, ..., 1, 0) = 1`.
// Note that this final query is in fact a constant that
// is independent from the proof. So we should avoid
// (de)serialize it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProductCheckSubClaim<F: PrimeField, ZC: ZeroCheck<F>> {
    // the SubClaim from the ZeroCheck
    pub zero_check_sub_claim: ZC::ZeroCheckSubClaim,
    // final query which consists of
    // - the vector `(1, ..., 1, 0)` (needs to be reversed because Arkwork's MLE uses big-endian
    //   format for points)
    // The expected final query evaluation is 1
    pub final_query: (Vec<F>, F),
    pub alpha: F,
}

/// A product check proof consists of
/// - a zerocheck proof
/// - a product polynomial commitment
/// - a polynomial commitment for the fractional polynomial
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProductCheckProof<
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    ZC: ZeroCheck<E::ScalarField>,
> {
    pub zero_check_proof: ZC::ZeroCheckProof,
    pub prod_x_comm: PCS::Commitment,
    pub frac_comm: PCS::Commitment,
}

/// A distributed version of ProductCheck protocol generates a product check proof for
/// the product of two lists of n-variate multilinear polynomials `(f1, f2, ..., fk)`
/// and `(g1, ..., gk)` satisfying:
///  \prod_{x \in {0,1}^n} f1(x) * ... * fk(x) = \prod_{x \in {0,1}^n} g1(x) * ... * gk(x)
///
/// The distributed version of ProductCheck gives a different proof structure than the
/// non-distributed version, because the Quark system for original product check isn't
/// easily distributed. The proof is different for the same problem when the number of
/// workers is different.
///
/// Prover steps:
/// 1. The worker provers build MLE `frac(x)` s.t. `frac(x) = f1(x) * ... * fk(x) / (g1(x) * ... * gk(x))`
/// distributedly, as each worker prover holds part of `f1, ... fk, g1, ..., gk`.
/// 2. The worker provers build MLE `prod_worker(x)` from their part of `frac(x)`.
/// They send_uniform the result of product to the master prover.
/// 3. The master prover builds MLE `prod_master(x)` from the product of `prod_worker(x)`s.
/// 4. All provers collaboratively compute the commitments for `frac(x)`, `prod_worker(x)`,
/// and `prod_master(x)`.
/// 5. The master prover submits the commitments to the transcript and generates two
/// challenges `alpha0` and `alpha1`.
/// 6. The provers collaboratively build the virtual polynomial `Q(x)` and generate
/// the zero-check proof.
///     Q(x) = frac(x) * g1(x) * ... * gk(x) - f1(x) * ... * fk(x)
///          + alpha0 * (prod_worker(x) - p1_worker(x) * p2_worker(x))
///          + alpha1 * (prod_master(x) - p1_master(x) * p2_master(x))
///     where
///         p1_master(x) = (1-x1) * prod_worker(x_2..t, 0, 1, ..., 1, 0) + x1 * prod_master(x_2..t, 0, x_t+1..n)
///         p2_master(x) = (1-x1) * prod_worker(x_2..t, 1, 0, ..., 0, 1) + x1 * prod_master(x_2..t, 1, x_t+1..n)
///         p1_worker(x) = (1-x_{t+1}) * frac(x_1..t, x_{t+2}..n, 0) + x_{t+1} * prod_worker(x_1..t, x_{t+2}..n, 0)
///         p2_worker(x) = (1-x_{t+1}) * frac(x_1..t, x_{t+2}..n, 1) + x_{t+1} * prod_worker(x_1..t, x_{t+2}..n, 1)
///         t = log2(num_workers)
/// 7. The master prover constructs the final proof for product check.
///
/// Verifier follows the same steps, with the new definition of the
/// virtual polynomial `Q(x)` in mind.
pub trait ProductCheckDistributed<E, PCS>: ZeroCheckDistributed<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<E>,
{
    type ProductCheckSubClaim;
    type ProductCheckProof;

    /// The master prover protocol of the distributed product check for proving
    /// the product of the polynomial over {0,1}^`num_vars`.
    #[allow(clippy::type_complexity)]
    fn prove_master(
        pcs_param_master: &PCS::MasterProverParam,
        num_polys: usize,
        num_vars: usize,
        transcript: &mut Self::Transcript,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<(Self::ProductCheckProof, Self::MultilinearExtension), PolyIOPErrors>;

    /// The worker prover protocol of the distributed product check for proving
    /// the product of the polynomial over {0,1}^`num_vars`.
    #[allow(clippy::type_complexity)]
    fn prove_worker(
        pcs_param_worker: &PCS::WorkerProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(Self::MultilinearExtension, Self::MultilinearExtension), PolyIOPErrors>;

    /// Verify a distributed product check proof and generate the subclaim
    /// for polynomial evaluations.
    fn verify(
        proof: &Self::ProductCheckProof,
        aux_info: &VPAuxInfo<E::ScalarField>,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ProductCheckSubClaim, PolyIOPErrors>;
}

/// A distributed version of product check subclaim consists of
/// - A zero check IOP subclaim for the virtual polynomial
/// - Two random challenges `alpha0` and `alpha1`
/// - A final query for `prod(1, ..., 1, 0, 1, ..., 1) = 1`.
///     - the `log_num_workers` term of final query is 0
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProductCheckSubClaimDistributed<F: PrimeField, ZC: ZeroCheckDistributed<F>> {
    pub zero_check_sub_claim: ZC::ZeroCheckSubClaim,
    pub final_query: (Vec<F>, F),
    pub alpha: (F, F),
}

/// A distributed version of product check proof consists of
/// - a zerocheck proof
/// - the result of the product polynomial
/// - the log number of workers
/// - a commitment for the product polynomial from the master prover
/// - a commitment for the product polynomial from the worker provers
/// - a commitment for the fractional polynomial
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProductCheckProofDistributed<
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<E>,
    ZC: ZeroCheckDistributed<E::ScalarField>,
> {
    pub zero_check_proof: ZC::ZeroCheckProof,
    pub result: <E as Pairing>::ScalarField,
    pub log_num_workers: usize,
    pub prod_master_comm: PCS::Commitment,
    pub prod_worker_comm: PCS::Commitment,
    pub frac_comm: PCS::Commitment,
}

impl<E, PCS> ProductCheck<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
{
    type ProductCheckSubClaim = ProductCheckSubClaim<E::ScalarField, Self>;
    type ProductCheckProof = ProductCheckProof<E, PCS, Self>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing ProductCheck transcript")
    }

    fn prove(
        pcs_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::ProductCheckProof,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "prod_check prove");

        if fxs.is_empty() {
            return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
        }
        if fxs.len() != gxs.len() {
            return Err(PolyIOPErrors::InvalidParameters(
                "fxs and gxs have different number of polynomials".to_string(),
            ));
        }
        for poly in fxs.iter().chain(gxs.iter()) {
            if poly.num_vars != fxs[0].num_vars {
                return Err(PolyIOPErrors::InvalidParameters(
                    "fx and gx have different number of variables".to_string(),
                ));
            }
        }

        // compute the fractional polynomial frac_p s.t.
        // frac_p(x) = f1(x) * ... * fk(x) / (g1(x) * ... * gk(x))
        let frac_poly = compute_frac_poly(fxs, gxs)?;
        // compute the product polynomial
        let prod_x = compute_product_poly(&frac_poly)?;

        // generate challenge
        let frac_comm = PCS::commit(pcs_param, &frac_poly)?;
        let prod_x_comm = PCS::commit(pcs_param, &prod_x)?;
        transcript.append_serializable_element(b"frac(x)", &frac_comm)?;
        transcript.append_serializable_element(b"prod(x)", &prod_x_comm)?;
        let alpha = transcript.get_and_append_challenge(b"alpha")?;

        // build the zero-check proof
        let (zero_check_proof, _) =
            prove_zero_check(fxs, gxs, &frac_poly, &prod_x, &alpha, transcript)?;

        end_timer!(start);

        Ok((
            ProductCheckProof {
                zero_check_proof,
                prod_x_comm,
                frac_comm,
            },
            prod_x,
            frac_poly,
        ))
    }

    fn verify(
        proof: &Self::ProductCheckProof,
        aux_info: &VPAuxInfo<E::ScalarField>,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ProductCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "prod_check verify");

        // update transcript and generate challenge
        transcript.append_serializable_element(b"frac(x)", &proof.frac_comm)?;
        transcript.append_serializable_element(b"prod(x)", &proof.prod_x_comm)?;
        let alpha = transcript.get_and_append_challenge(b"alpha")?;

        // invoke the zero check on the iop_proof
        // the virtual poly info for Q(x)
        let zero_check_sub_claim = <Self as ZeroCheck<E::ScalarField>>::verify(
            &proof.zero_check_proof,
            aux_info,
            transcript,
        )?;

        // the final query is on prod_x
        let mut final_query = vec![E::ScalarField::one(); aux_info.num_variables];
        // the point has to be reversed because Arkworks uses big-endian.
        final_query[0] = E::ScalarField::zero();
        let final_eval = E::ScalarField::one();

        end_timer!(start);

        Ok(ProductCheckSubClaim {
            zero_check_sub_claim,
            final_query: (final_query, final_eval),
            alpha,
        })
    }
}

impl<E, PCS> ProductCheckDistributed<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        MasterPolynomialHandle = usize,
        WorkerPolynomialHandle = Arc<DenseMultilinearExtension<E::ScalarField>>,
        ProverParam = MultilinearProverParam<E>,
        MasterProverParam = MultilinearProverParam<E>,
    >,
{
    type ProductCheckSubClaim = ProductCheckSubClaimDistributed<E::ScalarField, Self>;
    type ProductCheckProof = ProductCheckProofDistributed<E, PCS, Self>;

    fn prove_master(
        pcs_param_master: &PCS::MasterProverParam,
        num_polys: usize,
        num_vars: usize,
        transcript: &mut Self::Transcript,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<(Self::ProductCheckProof, Self::MultilinearExtension), PolyIOPErrors> {
        let start = start_timer!(|| "Distributed prod check; master");
        let log_num_workers = master_channel.log_num_workers();

        let preparation = start_timer!(|| "Distributed prod check preparation; master");
        master_channel.send_uniform(b"prod check starting signal")?;
        let sub_prod: Vec<E::ScalarField> = master_channel.recv()?;
        let prod_master = compute_product_poly(&Arc::new(
            DenseMultilinearExtension::from_evaluations_vec(log_num_workers, sub_prod.clone()),
        ))?;

        let frac_comm =
            PCS::commit_distributed_master(pcs_param_master, &num_vars, master_channel)?;
        let prod_worker_comm =
            PCS::commit_distributed_master(pcs_param_master, &num_vars, master_channel)?;
        let prod_master_comm = PCS::commit(pcs_param_master, &prod_master)?;

        transcript.append_serializable_element(b"frac(x)", &frac_comm)?;
        transcript.append_serializable_element(b"prod_worker(x)", &prod_worker_comm)?;
        transcript.append_serializable_element(b"prod_master(x)", &prod_master_comm)?;
        let alpha0 = transcript.get_and_append_challenge(b"alpha0")?;
        let alpha1 = transcript.get_and_append_challenge(b"alpha1")?;

        // build the zero-check proof
        //   frac(x) * g1(x) * ... * gk(x)
        // - f1(x) * ... * fk(x)
        // + alpha[0] * (
        //     prod_worker(x)
        //   - p1_worker(x) * p2_worker(x)
        // )
        // + alpha[1] * (
        //      prod_master(x)
        //   - p1_master(x) * p2_master(x)
        // )

        let mut p_evals_master = vec![vec![E::ScalarField::zero(); 2]; 1 << log_num_workers];
        for x in 0..1 << log_num_workers {
            let (x0, x1, sign) = get_index(x, log_num_workers);
            if !sign {
                p_evals_master[x][0] = sub_prod[x0];
                p_evals_master[x][1] = sub_prod[x1];
            } else {
                p_evals_master[x][0] = prod_master.evaluations[x0];
                p_evals_master[x][1] = prod_master.evaluations[x1];
            }
        }

        master_channel.send_different(p_evals_master)?;
        master_channel.send_uniform(&vec![alpha0, alpha1])?;
        let poly_aux_info = VPAuxInfo {
            max_degree: num_polys + 1,
            num_variables: num_vars,
            phantom: Default::default(),
        };
        let poly_products = vec![
            (alpha0, vec![0]),
            (-alpha0, vec![1, 2]),
            (alpha1, vec![3]),
            (-alpha1, vec![4, 5]),
            (E::ScalarField::one(), (6..7 + num_polys).collect()),
            (
                -E::ScalarField::one(),
                (7 + num_polys..7 + 2 * num_polys).collect(),
            ),
        ];
        end_timer!(preparation);

        let zero_check_proof = <Self as ZeroCheckDistributed<E::ScalarField>>::prove_master(
            &poly_aux_info,
            &poly_products,
            log_num_workers,
            transcript,
            master_channel,
        )?;

        end_timer!(start);

        Ok((
            ProductCheckProofDistributed {
                zero_check_proof,
                result: prod_master.evaluations[(1 << log_num_workers) - 2],
                log_num_workers,
                prod_master_comm,
                prod_worker_comm,
                frac_comm,
            },
            prod_master,
        ))
    }

    fn prove_worker(
        pcs_param_worker: &PCS::WorkerProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(Self::MultilinearExtension, Self::MultilinearExtension), PolyIOPErrors> {
        let start = start_timer!(|| "Distribution product check; prover");
        let preparation = start_timer!(|| "Distribution product check preparation; prover");

        if fxs.is_empty() {
            return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
        }
        if fxs.len() != gxs.len() {
            return Err(PolyIOPErrors::InvalidParameters(
                "fxs and gxs have different number of polynomials".to_string(),
            ));
        }

        let num_vars = fxs[0].num_vars;
        for poly in fxs.iter().chain(gxs.iter()) {
            if poly.num_vars != num_vars {
                return Err(PolyIOPErrors::InvalidParameters(
                    "fx and gx have different number of variables".to_string(),
                ));
            }
        }

        let start_data: [u8; 26] = worker_channel.recv()?;
        if &start_data != b"prod check starting signal" {
            return Err(PolyIOPErrors::InvalidDistributedMessage);
        }

        // compute the fractional polynomial frac_p s.t.
        // frac_p(x) = f1(x) * ... * fk(x) / (g1(x) * ... * gk(x))
        let frac_poly = compute_frac_poly(fxs, gxs)?;
        // compute the product polynomial
        let prod_worker = compute_product_poly(&frac_poly)?;

        worker_channel.send(&prod_worker.evaluations[(1 << num_vars) - 2])?;
        PCS::commit_distributed_worker(pcs_param_worker, &frac_poly, worker_channel)?;
        PCS::commit_distributed_worker(pcs_param_worker, &prod_worker, worker_channel)?;

        let p_master: Vec<E::ScalarField> = worker_channel.recv()?;
        let alpha: Vec<E::ScalarField> = worker_channel.recv()?;

        let mut p1_evals_worker = vec![E::ScalarField::zero(); 1 << num_vars];
        let mut p2_evals_worker = vec![E::ScalarField::zero(); 1 << num_vars];
        for x in 0..1 << num_vars {
            let (x0, x1, sign) = get_index(x, num_vars);
            if !sign {
                p1_evals_worker[x] = frac_poly.evaluations[x0];
                p2_evals_worker[x] = frac_poly.evaluations[x1];
            } else {
                p1_evals_worker[x] = prod_worker.evaluations[x0];
                p2_evals_worker[x] = prod_worker.evaluations[x1];
            }
        }

        let p1_worker = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            p1_evals_worker,
        ));

        let p2_worker = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            p2_evals_worker,
        ));

        let p1_master = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vec![p_master[0]; 1 << num_vars],
        ));

        let p2_master = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vec![p_master[1]; 1 << num_vars],
        ));

        let prod_master = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vec![p_master[0] * p_master[1]; 1 << num_vars],
        ));

        let aux_info = VPAuxInfo {
            max_degree: fxs.len() + 1,
            num_variables: num_vars,
            phantom: Default::default(),
        };

        let products = vec![
            (alpha[0], vec![0]),
            (-alpha[0], vec![1, 2]),
            (alpha[1], vec![3]),
            (-alpha[1], vec![4, 5]),
            (E::ScalarField::one(), (6..7 + fxs.len()).collect()),
            (
                -E::ScalarField::one(),
                (7 + fxs.len()..7 + 2 * fxs.len()).collect(),
            ),
        ];

        let poly = VirtualPolynomial::new_from_raw(
            aux_info,
            products,
            vec![
                prod_worker.clone(),
                p1_worker,
                p2_worker,
                prod_master,
                p1_master,
                p2_master,
                frac_poly.clone(),
            ]
            .into_iter()
            .chain(gxs.iter().cloned())
            .chain(fxs.iter().cloned())
            .collect(),
        );

        end_timer!(preparation);

        // build the zero-check proof
        <Self as ZeroCheckDistributed<E::ScalarField>>::prove_worker(&poly, worker_channel)?;

        end_timer!(start);

        Ok((prod_worker, frac_poly))
    }

    fn verify(
        proof: &Self::ProductCheckProof,
        aux_info: &VPAuxInfo<<E as Pairing>::ScalarField>,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ProductCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "Distributed prod check; verifier");

        // update transcript and generate challenge
        transcript.append_serializable_element(b"frac(x)", &proof.frac_comm)?;
        transcript.append_serializable_element(b"prod_worker(x)", &proof.prod_worker_comm)?;
        transcript.append_serializable_element(b"prod_master(x)", &proof.prod_master_comm)?;
        let alpha0 = transcript.get_and_append_challenge(b"alpha0")?;
        let alpha1 = transcript.get_and_append_challenge(b"alpha1")?;

        // invoke the zero check on the iop_proof
        // the virtual poly info for Q(x)
        let zero_check_sub_claim = <Self as ZeroCheck<E::ScalarField>>::verify(
            &proof.zero_check_proof,
            aux_info,
            transcript,
        )?;

        // the final query is on prod_x
        let mut final_query = vec![E::ScalarField::one(); aux_info.num_variables];
        // the point has to be reversed because Arkworks uses big-endian.
        final_query[aux_info.num_variables - proof.log_num_workers] = E::ScalarField::zero();
        let final_eval = proof.result;

        end_timer!(start);

        Ok(ProductCheckSubClaimDistributed {
            zero_check_sub_claim,
            final_query: (final_query, final_eval),
            alpha: (alpha0, alpha1),
        })
    }
}

#[cfg(test)]
mod test {
    use super::{ProductCheck, ProductCheckDistributed};
    use crate::{
        new_master_worker_channels,
        pcs::{
            prelude::MultilinearKzgPCS, PolynomialCommitmentScheme,
            PolynomialCommitmentSchemeDistributed,
        },
        poly_iop::{errors::PolyIOPErrors, PolyIOP},
        MultilinearProverParam,
    };
    use arithmetic::VPAuxInfo;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ec::pairing::Pairing;
    use ark_ff::batch_inversion;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use std::{marker::PhantomData, sync::Arc, thread::spawn};

    fn check_frac_poly<E>(
        frac_poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        fs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        gs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    ) where
        E: Pairing,
    {
        let mut flag = true;
        let num_vars = frac_poly.num_vars;
        for i in 0..1 << num_vars {
            let nom = fs
                .iter()
                .fold(E::ScalarField::from(1u8), |acc, f| acc * f.evaluations[i]);
            let denom = gs
                .iter()
                .fold(E::ScalarField::from(1u8), |acc, g| acc * g.evaluations[i]);
            if denom * frac_poly.evaluations[i] != nom {
                flag = false;
                break;
            }
        }
        assert!(flag);
    }
    // fs and gs are guaranteed to have the same product
    // fs and hs doesn't have the same product
    fn test_product_check_helper<E, PCS>(
        fs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        gs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        hs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        pcs_param: &PCS::ProverParam,
    ) -> Result<(), PolyIOPErrors>
    where
        E: Pairing,
        PCS: PolynomialCommitmentScheme<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        >,
    {
        let mut transcript = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        let (proof, prod_x, frac_poly) = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::prove(
            pcs_param,
            fs,
            gs,
            &mut transcript,
        )?;

        let mut transcript = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        // what's aux_info for?
        let aux_info = VPAuxInfo {
            max_degree: fs.len() + 1,
            num_variables: fs[0].num_vars,
            phantom: PhantomData::default(),
        };
        let prod_subclaim = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::verify(
            &proof,
            &aux_info,
            &mut transcript,
        )?;
        assert_eq!(
            prod_x.evaluate(&prod_subclaim.final_query.0).unwrap(),
            prod_subclaim.final_query.1,
            "different product"
        );
        check_frac_poly::<E>(&frac_poly, fs, gs);

        // bad path
        let mut transcript = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        let (bad_proof, prod_x_bad, frac_poly) = <PolyIOP<E::ScalarField> as ProductCheck<
            E,
            PCS,
        >>::prove(
            pcs_param, fs, hs, &mut transcript
        )?;

        let mut transcript = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let bad_subclaim = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::verify(
            &bad_proof,
            &aux_info,
            &mut transcript,
        )?;
        assert_ne!(
            prod_x_bad.evaluate(&bad_subclaim.final_query.0).unwrap(),
            bad_subclaim.final_query.1,
            "can't detect wrong proof"
        );
        // the frac_poly should still be computed correctly
        check_frac_poly::<E>(&frac_poly, fs, hs);

        Ok(())
    }

    fn test_product_check_helper_distributed<E, PCS>(
        fs: Vec<Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>>,
        gs: Vec<Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>>,
        log_num_workers: usize,
        pcs_param: PCS::ProverParam,
    ) -> Result<(), PolyIOPErrors>
    where
        E: Pairing,
        PCS: PolynomialCommitmentSchemeDistributed<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
            MasterPolynomialHandle = usize,
            WorkerPolynomialHandle = Arc<DenseMultilinearExtension<E::ScalarField>>,
            ProverParam = MultilinearProverParam<E>,
            WorkerProverParam = MultilinearProverParam<E>,
            MasterProverParam = MultilinearProverParam<E>,
        >,
    {
        let (pcs_param_master, pcs_param_worker) =
            PCS::prover_param_distributed(pcs_param, log_num_workers)?;

        let (mut master_channel, worker_channels) =
            new_master_worker_channels(true, log_num_workers, "127.0.0.1:0");
        // let (mut master_channel, worker_channels) = new_master_worker_thread_channels(log_num_workers);

        let mut transcript = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::init_transcript();
        let num_polys = fs[0].len();
        let num_worker_vars = fs[0][0].num_vars;
        let num_vars = log_num_workers + num_worker_vars;

        let handles = worker_channels
            .into_iter()
            .zip(pcs_param_worker.into_iter())
            .zip(fs.into_iter().zip(gs.into_iter()))
            .map(|((mut ch, pcs_param), (fs, gs))| {
                spawn(move || {
                    let (_, frac) =
                        <PolyIOP<E::ScalarField> as ProductCheckDistributed<E, PCS>>::prove_worker(
                            &pcs_param, &fs, &gs, &mut ch,
                        )?;
                    check_frac_poly::<E>(&frac, &fs, &gs);
                    Ok::<(), PolyIOPErrors>(())
                })
            })
            .collect::<Vec<_>>();

        let (proof, prod_master) =
            <PolyIOP<E::ScalarField> as ProductCheckDistributed<E, PCS>>::prove_master(
                &pcs_param_master,
                num_polys,
                num_vars,
                &mut transcript,
                &mut master_channel,
            )?;

        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Result<Vec<_>, _>>()?;

        let mut transcript = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::init_transcript();
        let subclaim = <PolyIOP<E::ScalarField> as ProductCheckDistributed<E, PCS>>::verify(
            &proof,
            &VPAuxInfo {
                max_degree: num_polys + 1,
                num_variables: num_vars,
                phantom: Default::default(),
            },
            &mut transcript,
        )?;

        assert_eq!(
            prod_master
                .evaluate(&subclaim.final_query.0[num_worker_vars..])
                .unwrap(),
            subclaim.final_query.1,
            "The final query evalution is not correct"
        );

        Ok(())
    }

    fn test_product_check(nv: usize) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        let f1: DenseMultilinearExtension<Fr> = DenseMultilinearExtension::rand(nv, &mut rng);
        let mut g1 = f1.clone();
        g1.evaluations.reverse();
        let f2: DenseMultilinearExtension<Fr> = DenseMultilinearExtension::rand(nv, &mut rng);
        let mut g2 = f2.clone();
        g2.evaluations.reverse();
        let fs = vec![Arc::new(f1), Arc::new(f2)];
        let gs = vec![Arc::new(g2), Arc::new(g1)];
        let mut hs = vec![];
        for _ in 0..fs.len() {
            hs.push(Arc::new(DenseMultilinearExtension::rand(
                fs[0].num_vars,
                &mut rng,
            )));
        }

        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bls12_381>::trim(&srs, None, Some(nv))?;

        test_product_check_helper::<Bls12_381, MultilinearKzgPCS<Bls12_381>>(
            &fs, &gs, &hs, &pcs_param,
        )?;

        Ok(())
    }

    fn test_product_check_distributed(
        num_polys: usize,
        nv: usize,
        log_num_workers: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        if nv < log_num_workers {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "nv should be greater than log_num_workers: {} < {}",
                nv, log_num_workers
            )));
        }

        let nv_worker = nv - log_num_workers;

        let fs: Vec<_> = (0..1 << log_num_workers)
            .map(|_| {
                (0..num_polys)
                    .map(|_| Arc::new(DenseMultilinearExtension::rand(nv_worker, &mut rng)))
                    .collect()
            })
            .collect();

        let gs: Vec<_> = (0..1 << log_num_workers)
            .map(|_| {
                (0..num_polys)
                    .map(|_| Arc::new(DenseMultilinearExtension::rand(nv_worker, &mut rng)))
                    .collect()
            })
            .collect();

        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bls12_381>::trim(&srs, None, Some(nv))?;

        let mut prod_f = fs
            .iter()
            .map(|f: &Vec<_>| {
                f.iter()
                    .map(|f| f.evaluations.iter().fold(Fr::from(1u128), |acc, x| acc * x))
                    .fold(Fr::from(1u128), |acc, x| acc * x)
            })
            .fold(Fr::from(1u128), |acc, x| acc * x);

        let prod_g = gs
            .iter()
            .map(|g: &Vec<_>| {
                g.iter()
                    .map(|g| g.evaluations.iter().fold(Fr::from(1u128), |acc, x| acc * x))
                    .fold(Fr::from(1u128), |acc, x| acc * x)
            })
            .fold(Fr::from(1u128), |acc, x| acc * x);

        let mut prod_g = [prod_g];

        batch_inversion(&mut prod_g);
        prod_f *= &prod_g[0];

        test_product_check_helper_distributed::<Bls12_381, MultilinearKzgPCS<Bls12_381>>(
            fs,
            gs,
            log_num_workers,
            pcs_param,
        )?;

        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        test_product_check(1)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        test_product_check(10)
    }
    #[test]
    fn test_polynomial_distributed() -> Result<(), PolyIOPErrors> {
        test_product_check_distributed(2, 7, 3)
    }
}
