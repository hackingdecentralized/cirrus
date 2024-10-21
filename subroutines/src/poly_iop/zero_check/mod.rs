// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the ZeroCheck protocol.

use std::{fmt::Debug, sync::Arc};

use crate::{poly_iop::{errors::PolyIOPErrors, sum_check::{SumCheck, SumCheckDistributed}, PolyIOP}, MasterProverChannel, WorkerProverChannel};
use arithmetic::{build_eq_x_r, eq_eval};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use transcript::IOPTranscript;

/// A zero check IOP subclaim for `f(x)` consists of the following:
///   - the initial challenge vector r which is used to build eq(x, r) in
///     SumCheck
///   - the random vector `v` to be evaluated
///   - the claimed evaluation of `f(v)`
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ZeroCheckSubClaim<F: PrimeField> {
    // the evaluation point
    pub point: Vec<F>,
    /// the expected evaluation
    pub expected_evaluation: F,
    // the initial challenge r which is used to build eq(x, r)
    pub init_challenge: Vec<F>,
}

/// A ZeroCheck for `f(x)` proves that `f(x) = 0` for all `x \in {0,1}^num_vars`
/// It is derived from SumCheck.
pub trait ZeroCheck<F: PrimeField>: SumCheck<F> {
    type ZeroCheckSubClaim: Clone + Debug + Default + PartialEq;
    type ZeroCheckProof: Clone + Debug + Default + PartialEq;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a ZeroCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// ZeroCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// initialize the prover to argue for the sum of polynomial over
    /// {0,1}^`num_vars` is zero.
    fn prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckProof, PolyIOPErrors>;

    /// verify the claimed sum using the proof
    fn verify(
        proof: &Self::ZeroCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckSubClaim, PolyIOPErrors>;
}

/// A distributed version of ZeroCheck Poly IOP generates zerocheck proof for
/// `f(x) = 0` for all `x \in {0,1}^num_vars`. The sumcheck protocol called
/// is distributed to accelerate the prover.
pub trait ZeroCheckDistributed<F: PrimeField>: ZeroCheck<F> + SumCheckDistributed<F> {
    /// The master prover interacts with the transcript and worker provers to
    /// generate the zero check proof.
    fn prove_master(
        poly_aux_info: &Self::VPAuxInfo,
        poly_products: &Vec<(F, Vec<usize>)>,
        log_num_workers: usize,
        transcript: &mut Self::Transcript,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<Self::ZeroCheckProof, PolyIOPErrors>;

    /// The worker prover interacts with the master prover to form part of
    /// the zero check proof.
    fn prove_worker(
        poly: &Self::VirtualPolynomial,
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(), PolyIOPErrors>;
}

impl<F: PrimeField> ZeroCheck<F> for PolyIOP<F> {
    type ZeroCheckSubClaim = ZeroCheckSubClaim<F>;
    type ZeroCheckProof = Self::SumCheckProof;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<F>::new(b"Initializing ZeroCheck transcript")
    }

    fn prove(
        poly: &Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "zero check prove");

        let length = poly.aux_info.num_variables;
        let r = transcript.get_and_append_challenge_vectors(b"0check r", length)?;
        let f_hat = poly.build_f_hat(r.as_ref())?;
        let res = <Self as SumCheck<F>>::prove(&f_hat, transcript);

        end_timer!(start);
        res
    }

    fn verify(
        proof: &Self::ZeroCheckProof,
        fx_aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::ZeroCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "zero check verify");

        // check that the sum is zero
        if proof.proofs[0].evaluations[0] + proof.proofs[0].evaluations[1] != F::zero() {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "zero check: sum {} is not zero",
                proof.proofs[0].evaluations[0] + proof.proofs[0].evaluations[1]
            )));
        }

        // generate `r` and pass it to the caller for correctness check
        let length = fx_aux_info.num_variables;
        let r = transcript.get_and_append_challenge_vectors(b"0check r", length)?;

        // hat_fx's max degree is increased by eq(x, r).degree() which is 1
        let mut hat_fx_aux_info = fx_aux_info.clone();
        hat_fx_aux_info.max_degree += 1;
        let sum_subclaim =
            <Self as SumCheck<F>>::verify(F::zero(), proof, &hat_fx_aux_info, transcript)?;

        // expected_eval = sumcheck.expect_eval/eq(v, r)
        // where v = sum_check_sub_claim.point
        let eq_x_r_eval = eq_eval(&sum_subclaim.point, &r)?;
        let expected_evaluation = sum_subclaim.expected_evaluation / eq_x_r_eval;

        end_timer!(start);
        Ok(ZeroCheckSubClaim {
            point: sum_subclaim.point,
            expected_evaluation,
            init_challenge: r,
        })
    }
}

impl<F: PrimeField> ZeroCheckDistributed<F> for PolyIOP<F> {
    fn prove_master(
        poly_aux_info: &Self::VPAuxInfo,
        poly_products: &Vec<(F, Vec<usize>)>,
        log_num_workers: usize,
        transcript: &mut Self::Transcript,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<Self::ZeroCheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "Distributed zero check; master");

        let length = poly_aux_info.num_variables;
        let mut r = transcript.get_and_append_challenge_vectors(b"0check r", length)?;

        let coeffs = {
            let x = r.split_off(length - log_num_workers);
            build_eq_x_r(&x)?.evaluations.clone()
        };

        // master_channel.send(b"zero check starting signal")?;
        master_channel.send(&r)?;
        master_channel.send_all(coeffs)?;

        master_channel.recv::<[u8; 32]>().unwrap().iter().map( |msg| {
            (msg == b"zero check preparation completed").then_some(()).ok_or(PolyIOPErrors::WorkerNotMatching)
        }).collect::<Result<Vec<_>, _>>()?;

        let poly_aux_info = Self::VPAuxInfo {
            max_degree: poly_aux_info.max_degree + 1,
            ..poly_aux_info.clone()
        };

        let num_mle = poly_products
            .iter()
            .map(|(_, indices)| indices.iter().max().unwrap_or(&0))
            .max().unwrap_or(&0).clone();

        let poly_products = {
            let mut x = poly_products.clone();
            x.iter_mut()
                .for_each(|(_, indices)| {
                    indices.push(num_mle + 1);
                });
            x
        };

        let res = <Self as SumCheckDistributed<F>>::prove_master(
            &poly_aux_info,
            &poly_products,
            log_num_workers,
            transcript,
            master_channel,
        )?;

        end_timer!(start);
        Ok(res)
    }

    fn prove_worker(
        poly: &Self::VirtualPolynomial,
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(), PolyIOPErrors> {
        let timer = start_timer!(|| "Distributed zero check; worker");

        let start_data: [u8; 26] = worker_channel.recv()?;
        if &start_data != b"zero check starting signal" {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "zero check: invalid starting signal {:?}",
                start_data
            )));
        }

        let r: Vec<F> = worker_channel.recv()?;
        let coeff: F = worker_channel.recv()?;
        if r.len() != poly.aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "zero check: invalid r length {}",
                r.len()
            )));
        }

        let eq_x_r = build_eq_x_r(&r)?;
        let eq_x_r = Arc::new(
            DenseMultilinearExtension::from_evaluations_vec(eq_x_r.num_vars,
                eq_x_r.evaluations.iter().map(|x| (*x) * coeff).collect()
            ));
        let poly = {
            let mut x = poly.clone();
            x.mul_by_mle(eq_x_r, F::one())?;
            x
        };

        worker_channel.send(b"zero check preparation completed")?;
        let res = <Self as SumCheckDistributed<F>>::prove_worker(&poly, worker_channel);

        end_timer!(timer);
        res
    }
}

#[cfg(test)]
mod test {

    use std::thread::spawn;

    use super::{ZeroCheck, ZeroCheckDistributed};
    use crate::{new_master_worker_thread_channels, poly_iop::{errors::PolyIOPErrors, PolyIOP}};
    use arithmetic::VirtualPolynomial;
    use ark_bls12_381::Fr;
    use ark_std::test_rng;

    fn test_zerocheck(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        {
            // good path: zero virtual poly
            let poly =
                VirtualPolynomial::rand_zero(nv, num_multiplicands_range, num_products, &mut rng)?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&poly, &mut transcript)?;

            let poly_info = poly.aux_info.clone();
            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let zero_subclaim =
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)?;
            assert!(
                poly.evaluate(&zero_subclaim.point)? == zero_subclaim.expected_evaluation,
                "wrong subclaim"
            );
        }

        {
            // bad path: random virtual poly whose sum is not zero
            let (poly, _sum) =
                VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&poly, &mut transcript)?;

            let poly_info = poly.aux_info.clone();
            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;

            assert!(
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)
                    .is_err()
            );
        }

        Ok(())
    }

    fn test_zerocheck_distributed(
        n_log_provers: usize,
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        assert!(n_log_provers <= nv, "log number of provers should be no larger than number of variables");

        let mut rng = test_rng();

        {
            // good path: zero virtual poly
            let poly =
                VirtualPolynomial::rand_zero(nv, num_multiplicands_range, num_products, &mut rng)?;
            let distributed_poly = poly.distribute(n_log_provers)?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;

            let (mut master_channel, worker_channels) = new_master_worker_thread_channels(n_log_provers);

            let worker_handles: Vec<_> = worker_channels.into_iter()
                .zip(distributed_poly)
                .map( |(ch, poly)| {
                    (ch, poly.aux_info, poly.products, poly.flattened_ml_extensions)
                })
                .map( |(mut ch, aux_info, products, mle)| {
                    spawn(move || {
                        let poly = VirtualPolynomial::new_from_raw(aux_info, products, mle);
                        <PolyIOP<Fr> as ZeroCheckDistributed<Fr>>::prove_worker(&poly, &mut ch)
                    })
                })
                .collect();

            let proof = <PolyIOP<Fr> as ZeroCheckDistributed<Fr>>::prove_master(
                &poly.aux_info,
                &poly.products,
                n_log_provers,
                &mut transcript,
                &mut master_channel,
            )?;

            worker_handles.into_iter().map( |h| h.join().unwrap()).collect::<Result<Vec<_>, _>>()?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let subclaim = <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly.aux_info, &mut transcript)?;

            assert!(
                poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
                "wrong subclaim"
            )
        }

        {
            // good path: zero virtual poly
            let (poly, _) =
                VirtualPolynomial::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
            let distributed_poly = poly.distribute(n_log_provers)?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;

            let (mut master_channel, worker_channels) = new_master_worker_thread_channels(n_log_provers);

            let worker_handles: Vec<_> = worker_channels.into_iter()
                .zip(distributed_poly)
                .map( |(mut ch, poly)| {
                    (ch, poly.aux_info, poly.products, poly.flattened_ml_extensions)
                })
                .map( |(mut ch, aux_info, products, mle)| {
                    spawn(move || {
                        let poly = VirtualPolynomial::new_from_raw(aux_info, products, mle);
                        <PolyIOP<Fr> as ZeroCheckDistributed<Fr>>::prove_worker(&poly, &mut ch)
                    })
                })
                .collect();

            let proof = <PolyIOP<Fr> as ZeroCheckDistributed<Fr>>::prove_master(
                &poly.aux_info,
                &poly.products,
                n_log_provers,
                &mut transcript,
                &mut master_channel,
            )?;

            worker_handles.into_iter().map( |h| h.join().unwrap()).collect::<Result<Vec<_>, _>>()?;

            let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;

            assert!(
                <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly.aux_info, &mut transcript)
                    .is_err()
            );
        }

        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 1;
        let num_multiplicands_range = (4, 5);
        let num_products = 1;

        test_zerocheck(nv, num_multiplicands_range, num_products)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 5;
        let num_multiplicands_range = (4, 9);
        let num_products = 5;

        let n_log_provers = 3;

        test_zerocheck(nv, num_multiplicands_range, num_products)?;
        test_zerocheck_distributed(n_log_provers, nv, num_multiplicands_range, num_products)
    }

    #[test]
    fn zero_polynomial_should_error() -> Result<(), PolyIOPErrors> {
        let nv = 0;
        let num_multiplicands_range = (4, 13);
        let num_products = 5;

        assert!(test_zerocheck(nv, num_multiplicands_range, num_products).is_err());
        Ok(())
    }
}
