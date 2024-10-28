// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the Permutation Check protocol

use self::util::computer_nums_and_denoms;
use crate::{
    pcs::{PolynomialCommitmentScheme, PolynomialCommitmentSchemeDistributed},
    poly_iop::{errors::PolyIOPErrors, prelude::ProductCheck, PolyIOP}, MasterProverChannel, MultilinearProverParam, WorkerProverChannel
};
use ark_ec::pairing::Pairing;
use ark_ff::One;
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use util::computer_nums_and_denoms_with_ids;
use std::sync::Arc;
use transcript::IOPTranscript;

use super::prod_check::ProductCheckDistributed;

/// A permutation subclaim consists of
/// - the SubClaim from the ProductCheck
/// - Challenges beta and gamma
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PermutationCheckSubClaim<E, PCS, PC>
where
    E: Pairing,
    PC: ProductCheck<E, PCS>,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// the SubClaim from the ProductCheck
    pub product_check_sub_claim: PC::ProductCheckSubClaim,
    /// Challenges beta and gamma
    pub challenges: (E::ScalarField, E::ScalarField),
}

pub mod util;

/// A PermutationCheck w.r.t. `(fs, gs, perms)`
/// proves that (g1, ..., gk) is a permutation of (f1, ..., fk) under
/// permutation `(p1, ..., pk)`
/// It is derived from ProductCheck.
///
/// A Permutation Check IOP takes the following steps:
///
/// Inputs:
/// - fs = (f1, ..., fk)
/// - gs = (g1, ..., gk)
/// - permutation oracles = (p1, ..., pk)
pub trait PermutationCheck<E, PCS>: ProductCheck<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type PermutationCheckSubClaim;
    type PermutationProof;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a PermutationCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// PermutationCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// Inputs:
    /// - fs = (f1, ..., fk)
    /// - gs = (g1, ..., gk)
    /// - permutation oracles = (p1, ..., pk)
    /// Outputs:
    /// - a permutation check proof proving that gs is a permutation of fs under
    ///   permutation
    /// - the product polynomial built during product check
    /// - the fractional polynomial built during product check
    ///
    /// Cost: O(N)
    #[allow(clippy::type_complexity)]
    fn prove(
        pcs_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::PermutationProof,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
        ),
        PolyIOPErrors,
    >;

    /// Verify that (g1, ..., gk) is a permutation of
    /// (f1, ..., fk) over the permutation oracles (perm1, ..., permk)
    fn verify(
        proof: &Self::PermutationProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors>;
}

/// A distributed version of PermutationCheck subclaim consists of
/// - a distributed version of product check subclaim
/// - challenges beta and gamma
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PermutationCheckSubClaimDistributed<E, PCS, PC>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<E>,
    PC: ProductCheckDistributed<E, PCS>,
{
    pub product_check_sub_claim: PC::ProductCheckSubClaim,
    pub challenges: (E::ScalarField, E::ScalarField),
}

/// Distributed permutation check IOP.
/// It provides the same functionality as the PermutationCheck protocol
/// but with its proof struct and subclaim struct a distributed version
/// as it calls the distributed version of the ProductCheck protocol.
pub trait PermutationCheckDistributed<E, PCS>: ProductCheckDistributed<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentSchemeDistributed<E>,
{
    type PermutationCheckSubClaim;
    type PermutationProof;

    /// Master prover protocol of the distributed permutation check. The master
    /// prover does not hold any polynomials at the beginning of the protocol.
    /// It interacts with the worker provers for the product check PIOP.
    /// 
    /// Outputs:
    /// - a distributed permutation check proof
    /// - the prod_master polynomial
    #[allow(clippy::type_complexity)]
    fn prove_master(
        pcs_param_master: &PCS::MasterProverParam,
        num_polys: usize,
        num_vars: usize,
        transcript: &mut Self::Transcript,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<(Self::PermutationProof, Self::MultilinearExtension), PolyIOPErrors>;

    /// Worker prover protocol of the distributed permutation check. The worker
    /// provers hold their part of the polynomials of (f1, ..., fk), (g1, ..., gk),
    /// (id1, ..., idk), and (perm1, ..., permk). 
    /// 
    /// Outputs:
    /// - the prod_worker polynomial
    /// - the frac polynomial
    #[allow(clippy::type_complexity)]
    fn prove_worker(
        pcs_param_worker: &PCS::WorkerProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        ids: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(Self::MultilinearExtension, Self::MultilinearExtension), PolyIOPErrors>;

    /// Verify a distributed version of permutation check proof and generate the
    /// corresponding subclaim.
    fn verify(
        proof: &Self::PermutationProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors>;
}


impl<E, PCS> PermutationCheck<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
{
    type PermutationCheckSubClaim = PermutationCheckSubClaim<E, PCS, Self>;
    type PermutationProof = Self::ProductCheckProof;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing PermutationCheck transcript")
    }

    fn prove(
        pcs_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::PermutationProof,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "Permutation check prove");
        if fxs.is_empty() {
            return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
        }
        if (fxs.len() != gxs.len()) || (fxs.len() != perms.len()) {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "fxs.len() = {}, gxs.len() = {}, perms.len() = {}",
                fxs.len(),
                gxs.len(),
                perms.len(),
            )));
        }

        let num_vars = fxs[0].num_vars;
        for ((fx, gx), perm) in fxs.iter().zip(gxs.iter()).zip(perms.iter()) {
            if (fx.num_vars != num_vars) || (gx.num_vars != num_vars) || (perm.num_vars != num_vars)
            {
                return Err(PolyIOPErrors::InvalidParameters(
                    "number of variables unmatched".to_string(),
                ));
            }
        }

        // generate challenge `beta` and `gamma` from current transcript
        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;
        let (numerators, denominators) = computer_nums_and_denoms(&beta, &gamma, fxs, gxs, perms)?;

        // invoke product check on numerator and denominator
        let (proof, prod_poly, frac_poly) = <Self as ProductCheck<E, PCS>>::prove(
            pcs_param,
            &numerators,
            &denominators,
            transcript,
        )?;

        end_timer!(start);
        Ok((proof, prod_poly, frac_poly))
    }

    fn verify(
        proof: &Self::PermutationProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "Permutation check verify");

        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        // invoke the zero check on the iop_proof
        let product_check_sub_claim =
            <Self as ProductCheck<E, PCS>>::verify(proof, aux_info, transcript)?;

        end_timer!(start);
        Ok(PermutationCheckSubClaim {
            product_check_sub_claim,
            challenges: (beta, gamma),
        })
    }
}

impl<E, PCS> PermutationCheckDistributed<E, PCS> for PolyIOP<E::ScalarField>
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
    type PermutationCheckSubClaim = PermutationCheckSubClaimDistributed<E, PCS, Self>;
    type PermutationProof = Self::ProductCheckProof;

    fn prove_master(
        pcs_param_master: &PCS::MasterProverParam,
        num_polys: usize,
        num_vars: usize,
        transcript: &mut Self::Transcript,
        master_channel: &mut impl MasterProverChannel,
    ) -> Result<(Self::PermutationProof, Self::MultilinearExtension), PolyIOPErrors> {
        let start = start_timer!(|| "Permutation check prove master");
        let log_num_workers = master_channel.log_num_workers();
        
        if num_vars < log_num_workers {
            return Err(PolyIOPErrors::InvalidParameters(format!(
                "num_vars = {} < log_num_workers = {}",
                num_vars, log_num_workers
            )));
        }
        master_channel.send_uniform(b"perm check starting signal")?;
        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        master_channel.send_uniform(&(beta, gamma))?;
        let (proof, prod_master) =
            <Self as ProductCheckDistributed<E, PCS>>::prove_master(
                pcs_param_master,
                num_polys,
                num_vars,
                transcript,
                master_channel,
            )?;

        end_timer!(start);
        Ok((proof, prod_master))
    }

    fn prove_worker(
        pcs_param_worker: &PCS::WorkerProverParam,
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        ids: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        worker_channel: &mut impl WorkerProverChannel,
    ) -> Result<(Self::MultilinearExtension, Self::MultilinearExtension), PolyIOPErrors> {
        let start = start_timer!(|| "Permutation check prove worker");
        if fxs.is_empty() {
            return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
        }
        if (fxs.len() != gxs.len()) || (fxs.len() != perms.len()) || (fxs.len() != ids.len()) {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "fxs.len() = {}, gxs.len() = {}, ids.len() = {}, perms.len() = {}",
                fxs.len(),
                gxs.len(),
                ids.len(),
                perms.len(),
            )));
        }

        let num_vars = fxs[0].num_vars;
        for ((fx, gx), (id, perm)) in fxs.iter().zip(gxs.iter()).zip(ids.iter().zip(perms.iter())) {
            if (fx.num_vars != num_vars)
                || (gx.num_vars != num_vars)
                || (id.num_vars != num_vars)
                || (perm.num_vars != num_vars)
            {
                return Err(PolyIOPErrors::InvalidParameters(
                    "number of variables unmatched".to_string(),
                ));
            }
        }

        let start_signal: [u8; 26] = worker_channel.recv()?;
        if start_signal != *b"perm check starting signal" {
            return Err(PolyIOPErrors::InvalidProof("invalid start signal".to_string()));
        }

        let (beta, gamma) = worker_channel.recv()?;
        let (numerators, denominators) = computer_nums_and_denoms_with_ids(&beta, &gamma, fxs, gxs, ids, perms)?;

        let (prod_worker, frac) = <Self as ProductCheckDistributed<E, PCS>>::prove_worker(
            pcs_param_worker,
            &numerators,
            &denominators,
            worker_channel,
        )?;

        end_timer!(start);
        Ok((prod_worker, frac))
    }

    fn verify(
        proof: &Self::PermutationProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "Permutation check verify");

        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        let product_check_sub_claim =
            <Self as ProductCheckDistributed<E, PCS>>::verify(proof, aux_info, transcript)?;

        if product_check_sub_claim.final_query.1 != E::ScalarField::one() {
            return Err(PolyIOPErrors::InvalidProof("final query is not one".to_string()));
        }

        end_timer!(start);
        Ok(PermutationCheckSubClaimDistributed {
            product_check_sub_claim,
            challenges: (beta, gamma),
        })
    }
}

#[cfg(test)]
mod test {
    use super::{PermutationCheck, PermutationCheckDistributed};
    use crate::{
        new_master_worker_thread_channels, new_master_worker_channels, pcs::{prelude::MultilinearKzgPCS, PolynomialCommitmentScheme, PolynomialCommitmentSchemeDistributed}, poly_iop::{errors::PolyIOPErrors, PolyIOP}, MultilinearProverParam
    };
    use arithmetic::{evaluate_opt, identity_permutation_mles, random_permutation_mles, random_permutation_with_corresponding_mles, split_into_chunks, transpose, VPAuxInfo};
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use std::{marker::PhantomData, sync::Arc, thread::spawn};

    type Kzg = MultilinearKzgPCS<Bls12_381>;

    fn test_permutation_check_helper<E, PCS>(
        pcs_param: &PCS::ProverParam,
        fxs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        gxs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        perms: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    ) -> Result<(), PolyIOPErrors>
    where
        E: Pairing,
        PCS: PolynomialCommitmentScheme<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        >,
    {
        let nv = fxs[0].num_vars;
        // what's AuxInfo used for?
        let poly_info = VPAuxInfo {
            max_degree: fxs.len() + 1,
            num_variables: nv,
            phantom: PhantomData::default(),
        };

        // prover
        let mut transcript =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let (proof, prod_x, _frac_poly) =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::prove(
                pcs_param,
                fxs,
                gxs,
                perms,
                &mut transcript,
            )?;

        // verifier
        let mut transcript =
            <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let perm_check_sub_claim = <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::verify(
            &proof,
            &poly_info,
            &mut transcript,
        )?;

        // check product subclaim
        if evaluate_opt(
            &prod_x,
            &perm_check_sub_claim.product_check_sub_claim.final_query.0,
        ) != perm_check_sub_claim.product_check_sub_claim.final_query.1
        {
            return Err(PolyIOPErrors::InvalidVerifier("wrong subclaim".to_string()));
        };

        Ok(())
    }

    fn test_permutation_check_distributed_helper<E, PCS>(
        pcs_param: PCS::ProverParam,
        log_num_workers: usize,
        fs: Vec<Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>>,
        gs: Vec<Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>>,
        ids: Vec<Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>>,
        perms: Vec<Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>>,
    ) -> Result<(), PolyIOPErrors>
    where
        E: Pairing,
        PCS: PolynomialCommitmentSchemeDistributed<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
            MasterPolynomialHandle = usize,
            WorkerPolynomialHandle = Arc<DenseMultilinearExtension<E::ScalarField>>,
            ProverParam = MultilinearProverParam<E>,
            MasterProverParam = MultilinearProverParam<E>,
            WorkerProverParam = MultilinearProverParam<E>,
        >,
    {
        let (pcs_param_master, pcs_param_worker) = PCS::prover_param_distributed(pcs_param, log_num_workers)?;
        // Define worker addresses as a Vec<&str>
        let worker_addrs: Vec<String> = (0..log_num_workers)
        .map(|i| format!("127.0.0.1:{}", 7879 + i))
        .collect();

        let worker_addrs_refs: Vec<&str> = worker_addrs.iter().map(|s| s.as_str()).collect();

        // Call new_master_worker_channels with use_sockets = true
        // let (mut master_channel, worker_channels) = new_master_worker_channels(true, log_num_workers,  "127.0.0.1:7878", worker_addrs_refs);
        
        let (mut master_channel, worker_channels) = new_master_worker_thread_channels(log_num_workers);
        let mut transcript = <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        let num_polys = fs[0].len();
        let num_vars = fs[0][0].num_vars + log_num_workers;

        let handles = worker_channels.into_iter().zip(pcs_param_worker.into_iter())
            .zip(fs.into_iter().zip(gs.into_iter()).zip(ids.into_iter().zip(perms.into_iter())))
            .map(|((mut ch, pcs_param), ((fs, gs), (ids, perms)))| {
                spawn(move || {
                    <PolyIOP<E::ScalarField> as PermutationCheckDistributed<E, PCS>>::prove_worker(
                        &pcs_param,
                        &fs,
                        &gs,
                        &ids,
                        &perms,
                        &mut ch,
                    )
                })
            }).collect::<Vec<_>>();

        let (proof, prod_master) = <PolyIOP<E::ScalarField> as PermutationCheckDistributed<E, PCS>>::prove_master(
            &pcs_param_master,
            num_polys,
            num_vars,
            &mut transcript,
            &mut master_channel,
        )?;

        handles.into_iter().map(|h| h.join().unwrap()).collect::<Result<Vec<_>, _>>()?;

        let mut transcript = <PolyIOP<E::ScalarField> as PermutationCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let aux_info = VPAuxInfo {
            max_degree: num_polys + 1,
            num_variables: num_vars,
            phantom: PhantomData::default(),
        };
        let perm_check_sub_claim = <PolyIOP<E::ScalarField> as PermutationCheckDistributed<E, PCS>>::verify(
            &proof,
            &aux_info,
            &mut transcript,
        )?;

        if evaluate_opt(
            &prod_master,
            &perm_check_sub_claim.product_check_sub_claim.final_query.0[num_vars-log_num_workers..],
        ) != perm_check_sub_claim.product_check_sub_claim.final_query.1
        {
            return Err(PolyIOPErrors::InvalidVerifier("wrong subclaim".to_string()));
        };

        Ok(())
    }

    fn test_permutation_check(nv: usize) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bls12_381>::trim(&srs, None, Some(nv))?;
        let id_perms = identity_permutation_mles(nv, 2);

        {
            // good path: (w1, w2) is a permutation of (w1, w2) itself under the identify
            // map
            let ws = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            // perms is the identity map
            test_permutation_check_helper::<Bls12_381, Kzg>(&pcs_param, &ws, &ws, &id_perms)?;
        }

        {
            // good path: f = (w1, w2) is a permutation of g = (w2, w1) itself under a map
            let mut fs = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            let gs = fs.clone();
            fs.reverse();
            // perms is the reverse identity map
            let mut perms = id_perms.clone();
            perms.reverse();
            test_permutation_check_helper::<Bls12_381, Kzg>(&pcs_param, &fs, &gs, &perms)?;
        }

        {
            // bad path 1: w is a not permutation of w itself under a random map
            let ws = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            // perms is a random map
            let perms = random_permutation_mles(nv, 2, &mut rng);

            assert!(
                test_permutation_check_helper::<Bls12_381, Kzg>(&pcs_param, &ws, &ws, &perms)
                    .is_err()
            );
        }

        {
            // bad path 2: f is a not permutation of g under a identity map
            let fs = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            let gs = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            // s_perm is the identity map

            assert!(test_permutation_check_helper::<Bls12_381, Kzg>(
                &pcs_param, &fs, &gs, &id_perms
            )
            .is_err());
        }

        Ok(())
    }

    fn test_permutation_check_distributed(nv: usize, log_num_workers: usize) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bls12_381>::trim(&srs, None, Some(nv))?;

        {
            let (ids, perms, fs) = random_permutation_with_corresponding_mles(nv, 3, &mut rng);
            let gs = fs.clone();

            let f = |x: Vec<_>| {
                let x = x.iter()
                    .map(|y| split_into_chunks(y, log_num_workers))
                    .collect::<Vec<_>>();
                transpose(x)
            };

            let (ids, perms, fs, gs) = (f(ids), f(perms), f(fs), f(gs));

            test_permutation_check_distributed_helper::<Bls12_381, Kzg>(
                pcs_param.clone(),
                log_num_workers,
                fs,
                gs,
                ids,
                perms,
            )?;
        }

        {
            let (ids, perms, fs) = random_permutation_with_corresponding_mles(nv, 3, &mut rng);
            let gs = fs.clone();

            let f = |x: Vec<_>| {
                let x = x.iter()
                    .map(|y| split_into_chunks(y, log_num_workers))
                    .collect::<Vec<_>>();
                transpose(x)
            };

            let (ids, perms, fs, gs) = (f(ids), f(perms), f(fs), f(gs));

            // bad path 1: fs is not a permutation of gs under a random map
            let mut perms = perms.clone();
            perms.reverse();
            assert!(
                test_permutation_check_distributed_helper::<Bls12_381, Kzg>(
                    pcs_param.clone(),
                    log_num_workers,
                    fs,
                    gs,
                    ids,
                    perms
                )
                .is_err()
            );
        }

        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        test_permutation_check(1)
    }

    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        test_permutation_check(5)?;
        test_permutation_check_distributed(5, 3)?;
        Ok(())
    }

    #[test]
    fn zero_polynomial_should_error() -> Result<(), PolyIOPErrors> {
        assert!(test_permutation_check(0).is_err());
        Ok(())
    }
}
