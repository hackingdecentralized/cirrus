use std::thread::spawn;
use std::sync::Arc;
use std::time::Instant;

use arithmetic::transpose;
use ark_poly::DenseMultilinearExtension;
use hyperplonk::prelude::WitnessColumn;
use hyperplonk::utils::build_f;
use ark_ec::pairing::Pairing;
use clap::Parser;
use hyperplonk::{errors::HyperPlonkErrors, prelude::MockCircuit, HyperPlonkSNARKDistributed};
use subroutines::{new_master_worker_thread_channels, MultilinearKzgPCS, PolyIOP, PolynomialCommitmentScheme};

use ark_bn254::Bn254;
use ark_bls12_381::Bls12_381;
use ark_bls12_377::Bls12_377;

#[derive(Parser)]
struct Args {
    #[clap(long, value_name = "number of threads", default_value = "8")]
    num_threads: usize,
    #[clap(
        long,
        value_name = "log number of workers (greater than 0)",
        default_value = "5"
    )]
    log_num_workers: usize,
    #[clap(
        long,
        value_name = "choose among [\"vanilla\", \"jellyfish\"]",
        default_value = "vanilla"
    )]
    gate: String,
    #[clap(long, value_name = "log number of constraints", default_value = "16")]
    log_num_constraints: usize,
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
        num_threads,
        log_num_workers,
        gate,
        log_num_constraints: nv,
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
        "bn254" => run_with_curve::<Bn254>(log_num_workers, gate, nv),
        "bls12_381" => run_with_curve::<Bls12_381>(log_num_workers, gate, nv),
        "bls12_377" => run_with_curve::<Bls12_377>(log_num_workers, gate, nv),
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
) -> Result<(), HyperPlonkErrors> {
    let gate = match gate.as_str() {
        "vanilla" => hyperplonk::custom_gate::CustomizedGates::vanilla_plonk_gate(),
        "jellyfish" => hyperplonk::custom_gate::CustomizedGates::jellyfish_turbo_plonk_gate(),
        _ => unreachable!(),
    };

    let mock_nv = nv - log_num_workers;
    let mock_log_num_workers = 1;
    let circuit = MockCircuit::<<E as Pairing>::ScalarField>::new(1 << mock_nv, &gate);

    let mut rng = ark_std::test_rng();
    let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, mock_nv)?;
    let ((pk_master, pk_workers), vk) = <PolyIOP<<E as Pairing>::ScalarField> as HyperPlonkSNARKDistributed<
        E,
        MultilinearKzgPCS<E>,
    >>::preprocess(
        &circuit.index, mock_log_num_workers, &pcs_srs
    )?;

    
    let (mut master_channel, worker_channels) =
        new_master_worker_thread_channels(mock_log_num_workers);
    let witnesses_distribution = circuit.witnesses
        .iter()
        .map(|w| {
            w.0.chunks(1 << (pk_master.params.num_variables()) - mock_log_num_workers)
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
                <PolyIOP<E::ScalarField> as HyperPlonkSNARKDistributed<
                    E,
                    MultilinearKzgPCS<E>,
                >>::prove_worker(&pk, &witness, &mut channel)
            })
        })
        .collect::<Vec<_>>();

    let proof = <PolyIOP<E::ScalarField> as HyperPlonkSNARKDistributed<
        E,
        MultilinearKzgPCS<E>,
    >>::prove_master(
        &pk_master,
        &circuit.public_inputs,
        mock_log_num_workers,
        &mut master_channel,
    )?;

    for handle in worker_handles {
        handle.join().unwrap()?;
    }

    // step 1: proof verification time
    #[cfg(feature = "print-time")]
    let start_verify = Instant::now();

    for _ in 0..(1<<log_num_workers) {
        let verify_result = <PolyIOP<E::ScalarField> as HyperPlonkSNARKDistributed<
            E,
            MultilinearKzgPCS<E>,
        >>::verify(&vk, &circuit.public_inputs, &proof)?;
        assert!(verify_result);
    }

    #[cfg(feature = "print-time")]
    println!(
        "[INFO] commitment opening verification time: {:?}",
        start_verify.elapsed()
    );

    // step 2: circuit multilinear extension evaluation
    let circuit = MockCircuit::<<E as Pairing>::ScalarField>::new(1 << nv, &gate);
    let selector_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> =
        circuit.index.selectors
            .iter()
            .map(|s| Arc::new(DenseMultilinearExtension::from(s)))
            .collect();
    let witness_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> = circuit.witnesses
        .iter()
        .map(|w| Arc::new(DenseMultilinearExtension::from(w)))
        .collect();

    #[cfg(feature = "print-time")]
    let start_circuit = Instant::now();
    let poly = build_f(
        &circuit.index.params.gate_func,
        nv,
        &selector_polys,
        &witness_polys
    )?;

    // we use (1, 1, \dots, 1) to mock the evaluation, because the time consumption
    // isn't related to point choice.
    poly.evaluate(&vec![E::ScalarField::from(1u64); nv])?;

    #[cfg(feature = "print-time")]
    println!("[INFO] permutation check time: {:?}", start_circuit.elapsed());

    Ok(())
}
