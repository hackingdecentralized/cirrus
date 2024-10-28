use ark_ec::pairing::Pairing;
use ark_std::test_rng;
use hyperplonk::{prelude::*, HyperPlonkSNARKDistributed};
use std::{thread::spawn, time::Instant};
use subroutines::{new_master_worker_thread_channels, MultilinearKzgPCS, MultilinearUniversalParams, PolyIOP, PolynomialCommitmentScheme};

type E = ark_bls12_381::Bls12_381;
type Fr = <E as Pairing>::ScalarField;

fn main() -> Result<(), HyperPlonkErrors> {
    let mut rng = test_rng();
    let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 24)?;

    for nv in (10..=24).step_by(2) {
        helper(nv, &pcs_srs)?;
    }

    Ok(())
}

fn helper(
    nv: usize,
    pcs_srs: &MultilinearUniversalParams<E>,
) -> Result<(), HyperPlonkErrors> {
    let log_num_workers = 1;

    let start = Instant::now();

    let gate= CustomizedGates::vanilla_plonk_gate();

    let circuit = MockCircuit::<Fr>::new(1 << nv, &gate);
    assert!(circuit.is_satisfied());
    let index = circuit.index;

    let (master_channel, worker_channels) = new_master_worker_thread_channels(log_num_workers);

    let ((pk_master, pk_workers), vk) =
        <PolyIOP::<Fr> as HyperPlonkSNARKDistributed<E, MultilinearKzgPCS<E>>>
        ::preprocess(&index, log_num_workers, &pcs_srs)?;

    let worker_handles = pk_workers
        .into_iter()
        .zip(worker_channels.into_iter())
        .map(|(pk, channel)| {
            spawn(move || {
                <PolyIOP<Fr> as HyperPlonkSNARKDistributed<E, MultilinearKzgPCS<E>>>::prove_worker(&pk, &channel)
            })
        }).collect::<Vec<_>>();

    let proof = <PolyIOP<Fr> as HyperPlonkSNARKDistributed<E, MultilinearKzgPCS<E>>>::prove_master(
        &pk_master,
        &circuit.public_inputs,
        &circuit.witnesses,
        log_num_workers,
        &master_channel,
    )?;

    for handle in worker_handles {
        handle.join().unwrap()?;
    }

    let elapsed = start.elapsed();
    println!("Proving time: {:?}", elapsed);

    assert!(
        <PolyIOP<Fr> as HyperPlonkSNARKDistributed<E, MultilinearKzgPCS<E>>>::verify(
            &vk, &circuit.public_inputs, &proof,
        )?
    );

    Ok(())
}