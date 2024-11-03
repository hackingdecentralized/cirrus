use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use hyperplonk::{
    errors::HyperPlonkErrors,
    structs::HyperPlonkProvingKeyWorker,
    HyperPlonkSNARKDistributed,
};
use std::{fs::File, path::PathBuf};
use subroutines::{
    MultilinearKzgPCS, PolyIOP, WorkerProverChannelSocket,
};

type E = ark_bls12_381::Bls12_381;
type Fr = <E as Pairing>::ScalarField;
type PCS = MultilinearKzgPCS<E>;

#[derive(Parser)]
struct Args {
    #[clap(long, value_name = "worker id", default_value = "0")]
    worker_id: usize,
    #[clap(
        long = "pk-worker",
        value_name = "worker proving key file path",
        default_value = "worker0.pk"
    )]
    pk_worker: PathBuf,
    #[clap(
        long,
        value_name = "master ip address",
        default_value = "127.0.0.0:9103"
    )]
    master_addr: String,
}

fn main() -> Result<(), HyperPlonkErrors> {
    let Args {
        worker_id,
        pk_worker,
        master_addr,
    } = Args::parse();

    if !pk_worker.exists() {
        return Err(HyperPlonkErrors::InvalidParameters(
            "worker proving key file does not exist".to_string(),
        ));
    }

    let file = File::open(pk_worker).unwrap();
    let pk_worker = HyperPlonkProvingKeyWorker::<E, PCS>::deserialize_compressed(&file).unwrap();

    let mut worker_channel = WorkerProverChannelSocket::bind(&master_addr, worker_id).unwrap();

    PolyIOP::<Fr>::prove_worker(&pk_worker, &mut worker_channel)
}
