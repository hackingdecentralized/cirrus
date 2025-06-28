use std::time::Instant;
use arithmetic::transpose;
use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use hyperplonk::{
    errors::HyperPlonkErrors,
    prelude::{MockCircuit, WitnessColumn},
    structs::{HyperPlonkProofDistributed, HyperPlonkProvingKeyMaster, HyperPlonkVerifyingKey},
    Cirrus,
};
use std::{fs::File, path::PathBuf};
use subroutines::{MasterProverChannel, MasterProverChannelSocket, MultilinearKzgPCS, PolyIOP};

// Import all the pairing-friendly curves
use ark_bn254::Bn254;
use ark_bls12_381::Bls12_381;
use ark_bls12_377::Bls12_377;

fn main() -> Result<(), HyperPlonkErrors> {
    Ok(())
}