use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_serialize::Write;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{test_rng, One, Zero};
use hyperplonk::{
    custom_gate::CustomizedGates,
    errors::HyperPlonkErrors,
    prelude::{SelectorColumn, WitnessColumn},
    structs::{HyperPlonkIndex, HyperPlonkParams, HyperPlonkProofDistributed},
    HyperPlonkSNARKDistributed,
};
use std::fs::File;
use subroutines::{DistributedError, MasterProverChannel, MasterProverChannelSocket};
use subroutines::{
    MultilinearKzgPCS, MultilinearUniversalParams, PolyIOP, PolynomialCommitmentScheme,
};

// Mock message struct for serialization/deserialization
#[derive(CanonicalSerialize, CanonicalDeserialize, PartialEq, Debug, Clone)]
struct TestMessage {
    data: u64,
}

fn main() -> Result<(), HyperPlonkErrors> {
    let srs_path: String = std::env::args().nth(1).expect("SRS path argument missing");
    run_master::<Bls12_381>(srs_path)
}

fn write_srs<E>(pcs_srs: &MultilinearUniversalParams<E>, path: String)
where
    E: Pairing,
{
    let mut f = File::create(path).unwrap();
    pcs_srs.serialize_uncompressed(&mut f).unwrap();
    f.flush().unwrap();
}

fn run_master<E>(srs_path: String) -> Result<(), HyperPlonkErrors>
where
    E: Pairing,
{
    let mut rng = test_rng();
    let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 16)?;

    write_srs::<E>(&pcs_srs, srs_path);

    let nv = 3 as usize;
    let num_constraints = 1 << nv;
    let num_pub_input = 4;
    let log_num_workers = 1;
    let master_addr = "127.0.0.1:7878";

    let gate_func = CustomizedGates {
        gates: vec![(1, Some(0), vec![0; 3]), (-1, None, vec![1])],
    };
    let params = HyperPlonkParams {
        num_constraints,
        num_pub_input,
        gate_func,
    };

    let w1 = WitnessColumn(
        (0..num_constraints)
            .map(|i| E::ScalarField::from((i % 4) as u128))
            .collect(),
    );
    let w2 = WitnessColumn(
        (0..num_constraints)
            .map(|i| i % 4)
            .map(|i| E::ScalarField::from((i * i * i) as u128))
            .collect(),
    );

    let permutation = vec![
        E::ScalarField::from(8u128),
        E::ScalarField::one(),
        E::ScalarField::from(2u128),
        E::ScalarField::from(3u128),
        E::ScalarField::zero(),
        E::ScalarField::from(5u128),
        E::ScalarField::from(6u128),
        E::ScalarField::from(7u128),
        E::ScalarField::from(4u128),
        E::ScalarField::from(9u128),
        E::ScalarField::from(10u128),
        E::ScalarField::from(15u128),
        E::ScalarField::from(12u128),
        E::ScalarField::from(13u128),
        E::ScalarField::from(14u128),
        E::ScalarField::from(11u128),
    ];

    let q = SelectorColumn(vec![E::ScalarField::one(); num_constraints]);
    let index = HyperPlonkIndex {
        params: params.clone(),
        permutation,
        selectors: vec![q],
    };

    let mut master_channel = MasterProverChannelSocket::new(log_num_workers);
    master_channel.connect_workers(master_addr)?;

    let ((pk_master, _), vk) =
        PolyIOP::<E::ScalarField>::preprocess(&index, log_num_workers, &pcs_srs)?;

    let proof: HyperPlonkProofDistributed<E, PolyIOP<E::ScalarField>, MultilinearKzgPCS<E>> =
        PolyIOP::<E::ScalarField>::prove_master(
            &pk_master,
            &w1.0[..num_pub_input],
            &[w1.clone(), w2],
            log_num_workers,
            &mut master_channel,
        )?;

    let verify_result = PolyIOP::<E::ScalarField>::verify(&vk, &w1.0[..num_pub_input], &proof)?;

    println!("Verification result: {}", verify_result);
    Ok(())
}

fn _test() -> Result<(), DistributedError> {
    let master_addr = "127.0.0.1:7878"; // Address to bind the master
    let log_num_workers = 1; // Set the number of workers as a power of two, e.g., 2^1 = 2 workers

    let mut master_channel = MasterProverChannelSocket::new(log_num_workers);

    // Connect to workers
    master_channel.connect_workers(master_addr)?;

    println!("Master connected to all workers.");

    // Example message to send
    let msg = TestMessage { data: 42 };

    // Send a uniform message to all workers
    master_channel.send_uniform(&msg)?;

    println!("Master sent uniform message to workers.");

    // Receive responses from all workers
    let responses: Vec<TestMessage> = master_channel.recv()?;

    println!("Master received responses from workers: {:?}", responses);

    Ok(())
}
