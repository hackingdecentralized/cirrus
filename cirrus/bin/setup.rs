use std::path::PathBuf;

use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalSerialize;
use clap::Parser;
use hyperplonk::{errors::HyperPlonkErrors, prelude::MockCircuit, HyperPlonkSNARKDistributed};
use subroutines::{MultilinearKzgPCS, PolyIOP, PolynomialCommitmentScheme};

type E = ark_bls12_381::Bls12_381;
type Fr = <E as Pairing>::ScalarField;

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
}

static MAX_NUM_VARS: usize = 30;

fn main() -> Result<(), HyperPlonkErrors> {
    let Args {
        log_num_workers,
        gate,
        log_num_constraints: nv,
        output,
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

    let mut rng = ark_std::test_rng();
    let circuit = MockCircuit::<Fr>::new(1 << nv, &gate);
    let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;

    let ((pk_master, pk_workers), vk) = <PolyIOP<Fr> as HyperPlonkSNARKDistributed<
        E,
        MultilinearKzgPCS<E>,
    >>::preprocess(
        &circuit.index, log_num_workers, &pcs_srs
    )?;

    println!(
        "[INFO] #constraints: {}, #witnesses: {}, ",
        1 << nv,
        (1 << nv) * circuit.num_witness_columns()
    );
    println!(
        "[INFO] Writing witnesses, proving keys and verification key to {}",
        output.display()
    );

    let path = output.join("circuit.plonk");
    let mut f = std::fs::File::create(path).unwrap();
    circuit.serialize_uncompressed(&mut f).unwrap();

    let path = output.join("master.pk");

    let mut f = std::fs::File::create(path).unwrap();
    pk_master.serialize_uncompressed(&mut f).unwrap();

    for (i, pk) in pk_workers.into_iter().enumerate() {
        let path = output.join(format!("worker_{}.pk", i));
        let mut f = std::fs::File::create(path).unwrap();
        pk.serialize_uncompressed(&mut f).unwrap();
    }

    let path = output.join("verify.key");
    let mut f = std::fs::File::create(path).unwrap();
    vk.serialize_uncompressed(&mut f).unwrap();

    println!("[INFO] Setup completed successfully");

    Ok(())
}
