use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_serialize::SerializationError;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use hyperplonk::{
    custom_gate::CustomizedGates,
    errors::HyperPlonkErrors,
    prelude::SelectorColumn,
    structs::{
        HyperPlonkIndex, HyperPlonkParams, HyperPlonkProvingKeyMaster, HyperPlonkProvingKeyWorker,
        HyperPlonkVerifyingKey,
    },
    HyperPlonkSNARKDistributed,
};
use std::fs::File;
use std::net::TcpStream;
use subroutines::{DistributedError, WorkerProverChannel, WorkerProverChannelSocket};
use subroutines::{MultilinearKzgPCS, MultilinearUniversalParams, PolyIOP};

// Mock message struct for serialization/deserialization
#[derive(CanonicalSerialize, CanonicalDeserialize, PartialEq, Debug, Clone)]
struct TestMessage {
    data: u64,
}

fn main() -> Result<(), HyperPlonkErrors> {
    let worker_id: usize = std::env::args()
        .nth(1)
        .expect("Worker ID argument missing")
        .parse()
        .expect("Worker ID must be a number");
    let srs_path: String = std::env::args().nth(2).expect("SRS path argument missing");
    run_worker::<Bls12_381>(srs_path, worker_id)
}

fn read_srs<E>(path: String) -> Result<MultilinearUniversalParams<E>, SerializationError>
where
    E: Pairing,
{
    let mut f = File::open(path)?;
    MultilinearUniversalParams::<E>::deserialize_uncompressed_unchecked(&mut f)
}

fn run_worker<E>(srs_path: String, worker_id: usize) -> Result<(), HyperPlonkErrors>
where
    E: Pairing,
{
    let pcs_srs = read_srs::<E>(srs_path).unwrap();

    let nv = 3 as usize;
    let num_constraints = 1 << nv;
    let num_pub_input = 4;
    let log_num_workers = 1;
    let master_addr = "127.0.0.1:7878";

    let gate_func = CustomizedGates {
        gates: vec![((false, 1), Some(0), vec![0; 3]), ((true, 1), None, vec![1])],
    };
    let params = HyperPlonkParams {
        num_constraints,
        num_pub_input,
        gate_func,
    };

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

    let proving_key: (
        (
            HyperPlonkProvingKeyMaster<E, MultilinearKzgPCS<E>>,
            Vec<HyperPlonkProvingKeyWorker<E, MultilinearKzgPCS<E>>>,
        ),
        HyperPlonkVerifyingKey<E, MultilinearKzgPCS<E>>,
    ) = PolyIOP::<E::ScalarField>::preprocess(&index, log_num_workers, &pcs_srs)?;

    let ((_, pk_workers), _) = proving_key;

    let socket = TcpStream::connect(master_addr).unwrap();
    let mut worker_channel = WorkerProverChannelSocket { worker_id, socket };

    PolyIOP::<E::ScalarField>::prove_worker(&pk_workers[worker_id], &mut worker_channel)
}

fn _test() -> Result<(), DistributedError> {
    let master_addr = "127.0.0.1:7878"; // Address to connect to the master
    let worker_id = std::env::args()
        .nth(1)
        .expect("Worker ID argument missing")
        .parse::<usize>()
        .expect("Invalid Worker ID");

    let socket =
        TcpStream::connect(master_addr).map_err(|_| DistributedError::WorkerConnectError)?;
    let mut worker_channel = WorkerProverChannelSocket { worker_id, socket };

    println!("Worker {} connected to master.", worker_id);

    // Receive a message from the master
    let received_msg: TestMessage = worker_channel.recv()?;
    println!(
        "Worker {} received message from master: {:?}",
        worker_id, received_msg
    );

    // Prepare a response
    let response = TestMessage {
        data: worker_id as u64 * 100,
    };

    // Send response back to the master
    worker_channel.send(&response)?;
    println!("Worker {} sent response back to master.", worker_id);

    Ok(())
}
