use ark_ec::pairing::Pairing;
use ark_std::test_rng;
use clap::Parser;
use hyperplonk::{
    errors::HyperPlonkErrors,
    prelude::{CustomizedGates, MockCircuit},
    structs::HyperPlonkProofDistributed,
    HyperPlonkSNARKDistributed,
};
use subroutines::{MasterProverChannelSocket, MultilinearKzgPCS, PolyIOP, PolynomialCommitmentScheme};

// Import all the pairing-friendly curves
use ark_bn254::Bn254;
use ark_bls12_381::Bls12_381;
use ark_bls12_377::Bls12_377;

#[derive(Parser)]
struct Args {
    #[clap(long, value_name = "number of threads", default_value = "1")]
    num_threads: usize,
    #[clap(long, value_name = "number of variables", default_value = "20")]
    num_vars: usize,
    #[clap(long, value_name = "log number of workers", default_value = "1")]
    log_num_workers: usize,
    #[clap(
        long,
        value_name = "choose curve among [\"bn254\", \"bls12_381\", \"bls12_377\", \"mnt4_753\", \"mnt6_753\"]",
        default_value = "bls12_381"
    )]
    curve: String,
}

fn main() -> Result<(), HyperPlonkErrors> {
    #[cfg(not(feature = "bench-master"))]
    {
        println!("Turn on the feature flag bench-master to run this binary");
        return Ok(());
    }

    #[cfg(feature = "bench-master")]
    {
        let Args {
            num_threads,
            num_vars,
            log_num_workers,
            curve,
        } = Args::parse();

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
            "bn254" => run_with_curve::<Bn254>(num_vars, log_num_workers),
            "bls12_381" => run_with_curve::<Bls12_381>(num_vars, log_num_workers),
            "bls12_377" => run_with_curve::<Bls12_377>(num_vars, log_num_workers),
            _ => {
                return Err(HyperPlonkErrors::InvalidParameters(
                    "curve should be one of [\"bn254\", \"bls12_381\", \"bls12_377\"]".to_string(),
                ));
            }
        }
    }
}

fn run_with_curve<E: Pairing>(
    num_vars: usize,
    log_num_workers: usize,
) -> Result<(), HyperPlonkErrors> {

    let mut rng = test_rng();
    let pcs_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, log_num_workers)?;
    let gate = CustomizedGates::vanilla_plonk_gate();
    let circuit = MockCircuit::<E::ScalarField>::new(1 << log_num_workers, &gate);
    assert!(circuit.is_satisfied());
    let index = circuit.index;

    let ((mut pk_master, _), _) = <PolyIOP<E::ScalarField> as HyperPlonkSNARKDistributed<
        E,
        MultilinearKzgPCS<E>,
    >>::preprocess(&index, log_num_workers, &pcs_srs)?;

    pk_master.log_num_workers = log_num_workers;
    pk_master.params.num_constraints = 1 << num_vars;

    let mut master_channel = MasterProverChannelSocket::new(log_num_workers);

    println!("[INFO] proving with {} variables", num_vars);

    // Prove the circuit
    let time = std::time::Instant::now();

    let _: HyperPlonkProofDistributed<E, PolyIOP<<E as Pairing>::ScalarField>, MultilinearKzgPCS<E>> =
        PolyIOP::<<E as Pairing>::ScalarField>::prove_master(
            &pk_master,
            &circuit.public_inputs,
            log_num_workers,
            &mut master_channel,
        )?;

    println!("[TIME] time elapsed: {:?}", time.elapsed());

    Ok(())
}
