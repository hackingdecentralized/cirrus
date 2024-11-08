use std::path::PathBuf;
use std::time::Instant;

use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalSerialize;
use clap::Parser;
use hyperplonk::{errors::HyperPlonkErrors, prelude::MockCircuit, HyperPlonkSNARKDistributed};
use subroutines::{MultilinearKzgPCS, PolyIOP, PolynomialCommitmentScheme};

use ark_bn254::Bn254;
use ark_bls12_381::Bls12_381;
use ark_bls12_377::Bls12_377;

#[derive(Parser)]
struct Args {
    #[clap(
        long,
        value_name = "log number of workers (greater than 0)",
        default_value = "1"
    )]
    log_num_workers: usize,
    #[clap(
        long,
        value_name = "choose among [\"vanilla\", \"jellyfish\"]",
        default_value = "vanilla"
    )]
    gate: String,
    #[clap(long, value_name = "log number of constraints", default_value = "1")]
    log_num_constraints: usize,
    #[clap(
        long,
        value_name = "directory path to witnesses, proving keys and the verification key",
        default_value = "./"
    )]
    output: PathBuf,
    #[clap(
        long,
        value_name = "choose curve among [\"bn254\", \"bls12_381\", \"bls12_377\"]",
        default_value = "bls12_381"
    )]
    curve: String,
}

static MAX_NUM_VARS: usize = 30;

fn main() -> Result<(), HyperPlonkErrors> {
    let Args {
        log_num_workers,
        gate,
        log_num_constraints: nv,
        output,
        curve,
    } = Args::parse();

    if log_num_workers > nv {
        return Err(HyperPlonkErrors::InvalidParameters(
            "log number of workers should be less than log number of constraints".to_string(),
        ));
    }

    if nv > MAX_NUM_VARS {
        return Err(HyperPlonkErrors::InvalidParameters(
            "number of constraints should be less than 2^32".to_string(),
        ));
    }

    if gate != "vanilla" && gate != "jellyfish" {
        return Err(HyperPlonkErrors::InvalidParameters(
            "gate should be either \"vanilla\" or \"jellyfish\"".to_string(),
        ));
    }

    if !output.exists() {
        return Err(HyperPlonkErrors::InvalidParameters(format!(
            "directory {} does not exist",
            output.display()
        )));
    }

    match curve.as_str() {
        "bn254" => run_with_curve::<Bn254>(log_num_workers, gate, nv, output),
        "bls12_381" => run_with_curve::<Bls12_381>(log_num_workers, gate, nv, output),
        "bls12_377" => run_with_curve::<Bls12_377>(log_num_workers, gate, nv, output),
        _ => {
            return Err(HyperPlonkErrors::InvalidParameters(
                "curve should be one of [\"bn254\", \"bls12_381\", \"bls12_377\", \"mnt4_753\", \"mnt6_753\"]".to_string(),
            ));
        }
    }
}

fn run_with_curve<E: Pairing>(
    log_num_workers: usize,
    gate: String,
    nv: usize,
    output: PathBuf,
) -> Result<(), HyperPlonkErrors> {
    println!("[INFO] Using curve: {}", std::any::type_name::<E>());

    #[cfg(feature = "print-time")]
    let start_total = Instant::now();
    println!(
        "[INFO] Generating proving keys and verification key for {} gate with {} constraints",
        gate,
        1 << nv
    );

    let gate = match gate.as_str() {
        "vanilla" => hyperplonk::custom_gate::CustomizedGates::vanilla_plonk_gate(),
        "jellyfish" => hyperplonk::custom_gate::CustomizedGates::jellyfish_turbo_plonk_gate(),
        _ => unreachable!(),
    };

    #[cfg(feature = "print-time")]
    let start_circuit = Instant::now();
    let circuit = MockCircuit::<<E as Pairing>::ScalarField>::new(1 << nv, &gate);
    #[cfg(feature = "print-time")]
    println!("[TIME] Circuit generation time: {:?}", start_circuit.elapsed());

    #[cfg(feature = "print-time")]
    let start_pcs = Instant::now();
    let mut rng = ark_std::test_rng();
    let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
    #[cfg(feature = "print-time")]
    println!("[TIME] PCS setup time: {:?}", start_pcs.elapsed());

    #[cfg(feature = "print-time")]
    let start_preprocess = Instant::now();
    let ((pk_master, pk_workers), vk) = <PolyIOP<<E as Pairing>::ScalarField> as HyperPlonkSNARKDistributed<
        E,
        MultilinearKzgPCS<E>,
    >>::preprocess(
        &circuit.index, log_num_workers, &pcs_srs
    )?;
    #[cfg(feature = "print-time")]
    println!("[TIME] Preprocessing time: {:?}", start_preprocess.elapsed());

    println!(
        "[INFO] #constraints: {}, #witnesses: {}, ",
        1 << nv,
        (1 << nv) * circuit.num_witness_columns()
    );

    #[cfg(feature = "print-time")]
    println!("[TIME] Total execution time: {:?}", start_total.elapsed());
    println!(
        "[INFO] Writing witnesses, proving keys and verification key to {}",
        output.display()
    );

    #[cfg(feature = "print-time")]
    let start_write_circuit = Instant::now();
    let path = output.join("circuit.plonk");
    let mut f = std::fs::File::create(path).unwrap();
    #[cfg(feature = "compress")]
    circuit.serialize_compressed(&mut f).unwrap();
    #[cfg(not(feature = "compress"))]
    circuit.serialize_uncompressed(&mut f).unwrap();
    #[cfg(feature = "print-time")]
    println!("[TIME] Writing circuit time: {:?}", start_write_circuit.elapsed());

    let path = output.join("master.pk");
    #[cfg(feature = "print-time")]
    let start_write_pk = Instant::now();
    let mut f = std::fs::File::create(path).unwrap();
    #[cfg(feature = "compress")]
    pk_master.serialize_compressed(&mut f).unwrap();
    #[cfg(not(feature = "compress"))]
    pk_master.serialize_uncompressed(&mut f).unwrap();

    for (i, pk) in pk_workers.into_iter().enumerate() {
        let path = output.join(format!("worker_{}.pk", i));
        let mut f = std::fs::File::create(path).unwrap();
        #[cfg(feature = "compress")]
        pk.serialize_compressed(&mut f).unwrap();
        #[cfg(not(feature = "compress"))]
        pk.serialize_uncompressed(&mut f).unwrap();
    }
    #[cfg(feature = "print-time")]
    println!("[TIME] Writing master pk time: {:?}", start_write_pk.elapsed());

    let path = output.join("verify.key");
    let mut f = std::fs::File::create(path).unwrap();
    #[cfg(feature = "compress")]
    vk.serialize_compressed(&mut f).unwrap();
    #[cfg(not(feature = "compress"))]
    vk.serialize_uncompressed(&mut f).unwrap();

    println!("[INFO] Setup completed successfully");

    Ok(())
}
