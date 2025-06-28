use arithmetic::transpose;
use ark_ec::pairing::Pairing;
use ark_std::test_rng;
use hyperplonk::{prelude::*, Cirrus};
use std::{thread::spawn, time::Instant};
use subroutines::{
    new_master_worker_channels, MultilinearKzgPCS, MultilinearUniversalParams, PolyIOP,
    PolynomialCommitmentScheme,
};

type E = ark_bls12_381::Bls12_381;
type Fr = <E as Pairing>::ScalarField;

fn main() -> Result<(), HyperPlonkErrors> {
    #[cfg(feature = "print-trace")]
    {
        println!("Benchmarking with feature print-trace isn't supported");
        return Ok(())
    }

    let mut rng = test_rng();
    let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 24)?;

    for nv in (20..=22).step_by(2) {
        helper(nv, &pcs_srs)?;
    }

    Ok(())
}

fn helper(nv: usize, pcs_srs: &MultilinearUniversalParams<E>) -> Result<(), HyperPlonkErrors> {
    let log_num_workers = 1;

    let gate = CustomizedGates::vanilla_plonk_gate();
    let circuit = MockCircuit::<Fr>::new(1 << nv, &gate);
    assert!(circuit.is_satisfied());
    let index = circuit.index;

    let (mut master_channel, worker_channels) =
        new_master_worker_channels(true, log_num_workers, "127.0.0.1:0");

    let ((pk_master, pk_workers), vk) = <PolyIOP<Fr> as Cirrus<
        E,
        MultilinearKzgPCS<E>,
    >>::preprocess(&index, log_num_workers, &pcs_srs)?;

    let witnesses_distribution = circuit.witnesses
        .iter()
        .map(|w| {
            w.0.chunks(1 << (nv - log_num_workers))
                .map(|chunk| WitnessColumn(chunk.to_vec()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let witnesses_distribution = transpose(witnesses_distribution);

    let worker_handles = pk_workers
        .into_iter()
        .zip(worker_channels.into_iter())
        .zip(witnesses_distribution.into_iter())
        .map(|((pk, mut channel), witness)| {
            spawn(move || {
                <PolyIOP<Fr> as Cirrus<E, MultilinearKzgPCS<E>>>::prove_worker(
                    &pk,
                    &witness,
                    &mut channel,
                )
            })
        })
        .collect::<Vec<_>>();

    let start = Instant::now();

    let proof = <PolyIOP<Fr> as Cirrus<E, MultilinearKzgPCS<E>>>::prove_master(
        &pk_master,
        &circuit.public_inputs,
        log_num_workers,
        &mut master_channel,
    )?;

    for handle in worker_handles {
        handle.join().unwrap()?;
    }

    let elapsed = start.elapsed();
    println!("Proving time: {:?}", elapsed);

    assert!(<PolyIOP<Fr> as Cirrus<
        E,
        MultilinearKzgPCS<E>,
    >>::verify(&vk, &circuit.public_inputs, &proof,)?);

    Ok(())
}
