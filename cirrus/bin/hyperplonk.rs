use std::time::Instant;

use ark_ec::pairing::Pairing;
use ark_std::test_rng;
use clap::Parser;
use hyperplonk::{
    prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
    HyperPlonkSNARK,
};
use subroutines::{
    pcs::{
        prelude::{MultilinearKzgPCS, MultilinearUniversalParams},
        PolynomialCommitmentScheme,
    },
    poly_iop::PolyIOP,
};

// Import all the pairing-friendly curves
use ark_bn254::Bn254;
use ark_bls12_381::Bls12_381;
use ark_bls12_377::Bls12_377;

#[derive(Parser)]
struct Args {
    #[clap(
        long,
        value_name = "choose among [\"bn254\", \"bls12_381\", \"bls12_377\"]",
        default_value = "bls12_381"
    )]
    curve: String,
    #[clap(
        long,
        value_name = "log number of variables",
        default_value = "16"
    )]
    log_num_vars: usize,
}

fn main() -> Result<(), HyperPlonkErrors> {
    let Args { curve, log_num_vars } = Args::parse();

    let thread = rayon::current_num_threads();
    println!("start benchmark with #{} threads", thread);
    let mut rng = test_rng();

    match curve.as_str() {
        "bn254" => run_with_curve::<Bn254>(log_num_vars, &mut rng)?,
        "bls12_381" => run_with_curve::<Bls12_381>(log_num_vars, &mut rng)?,
        "bls12_377" => run_with_curve::<Bls12_377>(log_num_vars, &mut rng)?,
        _ => {
            return Err(HyperPlonkErrors::InvalidParameters(
                "curve should be one of [\"bn254\", \"bls12_381\", \"bls12_377\"]".to_string(),
            ));
        }
    }

    Ok(())
}

fn run_with_curve<E: Pairing>(log_num_vars: usize, rng: &mut impl ark_std::rand::Rng) -> Result<(), HyperPlonkErrors> {
    let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(rng, log_num_vars)?;
    bench_vanilla_plonk::<E>(&pcs_srs, log_num_vars)?;
    Ok(())
}

fn bench_vanilla_plonk<E: Pairing>(
    pcs_srs: &MultilinearUniversalParams<E>,
    log_num_vars: usize,
) -> Result<(), HyperPlonkErrors> {
    println!("[INFO] Using curve: {}", std::any::type_name::<E>());

    let vanilla_gate = CustomizedGates::vanilla_plonk_gate();
    bench_mock_circuit_zkp_helper::<E>(log_num_vars, &vanilla_gate, pcs_srs)?;

    Ok(())
}

fn bench_mock_circuit_zkp_helper<E: Pairing>(
    log_num_vars: usize,
    gate: &CustomizedGates,
    pcs_srs: &MultilinearUniversalParams<E>,
) -> Result<(), HyperPlonkErrors> {
    // type Fr = <E as Pairing>::ScalarField;

    //==========================================================
    let circuit = MockCircuit::< <E as Pairing>::ScalarField>::new(1 << log_num_vars, gate);
    assert!(circuit.is_satisfied());
    let index = circuit.index;
    //==========================================================
    // generate pk and vks
    let start = Instant::now();
    let (_pk, _vk) = <PolyIOP<<E as Pairing>::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
        &index, pcs_srs,
    )?;
    println!(
        "[INFO] key extraction for {:?} variables: {:?} us",
        log_num_vars,
        start.elapsed()
    );
    let (pk, vk) = <PolyIOP<<E as Pairing>::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
        &index, pcs_srs,
    )?;
    //==========================================================
    // generate a proof
    let start = Instant::now();
    let _proof = <PolyIOP<<E as Pairing>::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
        &pk,
        &circuit.public_inputs,
        &circuit.witnesses,
    )?;
    let t = start.elapsed();
    println!(
        "[INFO] proving for {:?} variables: {:?} s",
        log_num_vars,
        t
    );

    let proof = <PolyIOP<<E as Pairing>::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
        &pk,
        &circuit.public_inputs,
        &circuit.witnesses,
    )?;
    //==========================================================
    // verify a proof
    let start = Instant::now();
    let verify = <PolyIOP<<E as Pairing>::ScalarField> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::verify(
        &vk,
        &circuit.public_inputs,
        &proof,
    )?;
    assert!(verify);
    println!(
        "[INFO] verifying for {:?} variables: {:?} us",
        log_num_vars,
        start.elapsed()
    );
    Ok(())
}
