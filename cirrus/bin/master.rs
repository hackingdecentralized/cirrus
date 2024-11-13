use std::time::Instant;
use arithmetic::transpose;
use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use hyperplonk::{
    errors::HyperPlonkErrors,
    prelude::{MockCircuit, WitnessColumn},
    structs::{HyperPlonkProofDistributed, HyperPlonkProvingKeyMaster, HyperPlonkVerifyingKey},
    HyperPlonkSNARKDistributed,
};
use std::{fs::File, path::PathBuf};
use subroutines::{MasterProverChannel, MasterProverChannelSocket, MultilinearKzgPCS, PolyIOP};

// Import all the pairing-friendly curves
use ark_bn254::Bn254;
use ark_bls12_381::Bls12_381;
use ark_bls12_377::Bls12_377;

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
        circuit_file,
        pk_master,
        verification_key,
        master_addr,
        curve,
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

    match curve.as_str() {
        "bn254" => run_with_curve::<Bn254>(circuit_file, pk_master, verification_key, master_addr),
        "bls12_381" => run_with_curve::<Bls12_381>(circuit_file, pk_master, verification_key, master_addr),
        "bls12_377" => run_with_curve::<Bls12_377>(circuit_file, pk_master, verification_key, master_addr),
        _ => {
            return Err(HyperPlonkErrors::InvalidParameters(
                "curve should be one of [\"bn254\", \"bls12_381\", \"bls12_377\"]".to_string(),
            ));
        }
    }
}

fn run_with_curve<E: Pairing>(
    circuit_file: PathBuf,
    pk_master_path: PathBuf,
    verification_key_path: PathBuf,
    master_addr: String,
) -> Result<(), HyperPlonkErrors> {

    #[cfg(feature = "print-time")]
    let start_read_circuit = Instant::now();
    let file = File::open(circuit_file).unwrap();
    #[cfg(feature = "compress")]
    let circuit = MockCircuit::<Fr>::deserialize_compressed(file).unwrap();
    #[cfg(not(feature = "compress"))]
    let circuit = MockCircuit::<<E as Pairing>::ScalarField>::deserialize_uncompressed(file).unwrap();
    #[cfg(feature = "print-time")]
    println!("[INFO] read circuit time: {:?}", start_read_circuit.elapsed());

    #[cfg(feature = "print-time")]
    let start_read_pk = Instant::now();
    let file = File::open(pk_master_path).unwrap();
    #[cfg(feature = "compress")]
    let pk_master = HyperPlonkProvingKeyMaster::<E, MultilinearKzgPCS::<E>>::deserialize_compressed(file).unwrap();
    #[cfg(not(feature = "compress"))]
    let pk_master = HyperPlonkProvingKeyMaster::<E, MultilinearKzgPCS::<E>>::deserialize_uncompressed(file).unwrap();
    #[cfg(feature = "print-time")]
    println!("[INFO] read pk time: {:?}", start_read_pk.elapsed());

    let log_num_workers = pk_master.log_num_workers;

    let file = File::open(verification_key_path).unwrap();
    #[cfg(feature = "compress")]
    let vk = HyperPlonkVerifyingKey::<E, MultilinearKzgPCS::<E>>::deserialize_compressed(file).unwrap();
    #[cfg(not(feature = "compress"))]
    let vk = HyperPlonkVerifyingKey::<E, MultilinearKzgPCS::<E>>::deserialize_uncompressed(file).unwrap();

    println!("[INFO] loaded circuit, master proving key and verification key");

    let mut master_channel = MasterProverChannelSocket::new(log_num_workers);
    master_channel.connect_workers(&master_addr)?;

    // Distribute the witnesses to the workers
    let witnesses_distribution = circuit.witnesses
        .iter()
        .map(|w| {
            w.0.chunks(1 << (pk_master.params.num_variables() - log_num_workers))
                .map(|chunk| WitnessColumn(chunk.to_vec()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let witnesses_distribution = transpose(witnesses_distribution);
    master_channel.send_different(witnesses_distribution)?;

    // Prove the circuit
    let time = std::time::Instant::now();

    let proof: HyperPlonkProofDistributed<E, PolyIOP<<E as Pairing>::ScalarField>, MultilinearKzgPCS<E>> =
        PolyIOP::<<E as Pairing>::ScalarField>::prove_master(
            &pk_master,
            &circuit.public_inputs,
            log_num_workers,
            &mut master_channel,
        )?;

    let verify_result = PolyIOP::<<E as Pairing>::ScalarField>::verify(&vk, &circuit.public_inputs, &proof)?;

    println!("[INFO] verification result: {}", verify_result);
    println!("[TIME] time elapsed: {:?}", time.elapsed());

    Ok(())
}
