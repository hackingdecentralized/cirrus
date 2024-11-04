use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use hyperplonk::{
    errors::HyperPlonkErrors,
    prelude::MockCircuit,
    structs::{HyperPlonkProofDistributed, HyperPlonkProvingKeyMaster, HyperPlonkVerifyingKey},
    HyperPlonkSNARKDistributed,
};
use std::{fs::File, path::PathBuf};
use subroutines::{MasterProverChannelSocket, MultilinearKzgPCS, PolyIOP};

type E = ark_bls12_381::Bls12_381;
type Fr = <E as Pairing>::ScalarField;
type PCS = MultilinearKzgPCS<E>;

#[derive(Parser)]
struct Args {
    #[clap(long, value_name = "number of threads", default_value = "1")]
    num_threads: usize,
    #[clap(
        long,
        value_name = "circuit witnesses and selectors file",
        default_value = "circuit.plonk"
    )]
    circuit_file: PathBuf,
    #[clap(
        long,
        value_name = "master proving key file path",
        default_value = "master.pk"
    )]
    pk_master: PathBuf,
    #[clap(
        long,
        value_name = "verification key file path",
        default_value = "verification.key"
    )]
    verification_key: PathBuf,
    #[clap(
        long,
        value_name = "master ip address",
        default_value = "127.0.0.0:9103"
    )]
    master_addr: String,
}

fn main() -> Result<(), HyperPlonkErrors> {
    let Args {
        num_threads,
        circuit_file,
        pk_master,
        verification_key,
        master_addr,
    } = Args::parse();

    if !circuit_file.exists() {
        return Err(HyperPlonkErrors::InvalidParameters(
            "circuit file does not exist".to_string(),
        ));
    }

    if !pk_master.exists() {
        return Err(HyperPlonkErrors::InvalidParameters(
            "master proving key file does not exist".to_string(),
        ));
    }

    if !verification_key.exists() {
        return Err(HyperPlonkErrors::InvalidParameters(
            "verification key file does not exist".to_string(),
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
    println!("[WARN] parallel feature is not enabled, using single thread");

    let file = File::open(circuit_file).unwrap();
    let circuit = MockCircuit::<Fr>::deserialize_uncompressed(file).unwrap();
    let file = File::open(pk_master).unwrap();
    let pk_master = HyperPlonkProvingKeyMaster::<E, PCS>::deserialize_uncompressed(file).unwrap();
    let log_num_workers = pk_master.log_num_workers;

    let file = File::open(verification_key).unwrap();
    let vk = HyperPlonkVerifyingKey::<E, PCS>::deserialize_uncompressed(file).unwrap();

    println!("[INFO] loaded circuit, master proving key and verification key");

    let mut master_channel = MasterProverChannelSocket::new(log_num_workers);
    master_channel.connect_workers(&master_addr)?;

    let time = std::time::Instant::now();

    let proof: HyperPlonkProofDistributed<E, PolyIOP<Fr>, MultilinearKzgPCS<E>> =
        PolyIOP::<Fr>::prove_master(
            &pk_master,
            &circuit.public_inputs,
            &circuit.witnesses,
            log_num_workers,
            &mut master_channel,
        )?;

    let verify_result = PolyIOP::<Fr>::verify(&vk, &circuit.public_inputs, &proof)?;

    println!("[INFO] verification result: {}", verify_result);
    println!("[INFO] time elapsed: {:?}", time.elapsed());

    Ok(())
}
