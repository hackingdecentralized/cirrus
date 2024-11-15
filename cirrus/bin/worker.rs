use std::time::Instant;
use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use hyperplonk::{
    errors::HyperPlonkErrors, prelude::WitnessColumn, structs::HyperPlonkProvingKeyWorker, HyperPlonkSNARKDistributed
};
use std::{fs::File, path::PathBuf};
use subroutines::{MultilinearKzgPCS, PolyIOP, WorkerProverChannel, WorkerProverChannelSocket};

// Import all the pairing-friendly curves
use ark_bn254::Bn254;
use ark_bls12_381::Bls12_381;
use ark_bls12_377::Bls12_377;

#[derive(Parser)]
struct Args {
    #[clap(long, value_name = "number of threads", default_value = "1")]
    num_threads: usize,
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
    #[clap(
        long,
        value_name = "choose curve among [\"bn254\", \"bls12_381\", \"bls12_377\", \"mnt4_753\", \"mnt6_753\"]",
        default_value = "bls12_381"
    )]
    curve: String,
}

fn main() -> Result<(), HyperPlonkErrors> {
    let Args {
        num_threads,
        worker_id,
        pk_worker,
        master_addr,
        curve,
    } = Args::parse();

    if !pk_worker.exists() {
        return Err(HyperPlonkErrors::InvalidParameters(
            "worker proving key file does not exist".to_string(),
        ));
    }

    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    #[cfg(feature = "parallel")]
    println!("[INFO] rayon threads: {:?}", rayon::current_num_threads());

    #[cfg(not(feature = "parallel"))]
    println!("[WARN] parallel feature is disabled, using single thread");

    match curve.as_str() {
        "bn254" => run_with_curve::<Bn254>(worker_id, pk_worker, master_addr),
        "bls12_381" => run_with_curve::<Bls12_381>(worker_id, pk_worker, master_addr),
        "bls12_377" => run_with_curve::<Bls12_377>(worker_id, pk_worker, master_addr),
        _ => {
            return Err(HyperPlonkErrors::InvalidParameters(
                "curve should be one of [\"bn254\", \"bls12_381\", \"bls12_377\"]".to_string(),
            ));
        }
    }
}

fn run_with_curve<E: Pairing>(
    worker_id: usize,
    pk_worker_path: PathBuf,
    master_addr: String,
) -> Result<(), HyperPlonkErrors> {
    #[cfg(feature = "print-time")]
    let start_read_pk = Instant::now();
    let file = File::open(pk_worker_path).unwrap();
    #[cfg(feature = "compress")]
    let pk_worker = HyperPlonkProvingKeyWorker::<E, MultilinearKzgPCS::<E>>::deserialize_compressed(&file).unwrap();
    #[cfg(not(feature = "compress"))]
    let pk_worker = HyperPlonkProvingKeyWorker::<E, MultilinearKzgPCS::<E>>::deserialize_uncompressed(&file).unwrap();
    #[cfg(feature = "print-time")]
    println!("[INFO] read worker proving key in {:?}", start_read_pk.elapsed());

    let mut worker_channel = WorkerProverChannelSocket::bind(&master_addr, worker_id).unwrap();

    let witness: Vec<WitnessColumn<_>> = worker_channel.recv()?;
    PolyIOP::<<E as Pairing>::ScalarField>::prove_worker(&pk_worker, &witness, &mut worker_channel)
}
