use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use errors::DistributedError;

mod channel_enum;
mod errors;
mod network_channel;
pub mod prelude;
mod thread_channel;

pub trait MasterProverChannel {
    /// TODO
    fn send_uniform(&mut self, msg: &impl CanonicalSerialize) -> Result<(), DistributedError>;

    /// TODO
    fn send_different<T: CanonicalSerialize + Send>(
        &mut self,
        msg: Vec<T>,
    ) -> Result<(), DistributedError>;

    /// TODO
    fn recv<T: CanonicalDeserialize + Send>(&mut self) -> Result<Vec<T>, DistributedError>;

    /// TODO
    fn log_num_workers(&self) -> usize;
}

pub trait WorkerProverChannel {
    /// TODO
    fn send(&mut self, msg: &(impl CanonicalSerialize + Send)) -> Result<(), DistributedError>;

    /// TODO
    fn recv<T: CanonicalDeserialize>(&mut self) -> Result<T, DistributedError>;

    /// TODO
    fn worker_id(&self) -> usize;
}

#[cfg(test)]
mod test {
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
    use ark_std::UniformRand;

    #[test]
    fn serialize() -> Result<(), SerializationError> {
        let mut rng = ark_std::test_rng();

        let a = <Bls12_381 as Pairing>::G1::rand(&mut rng);

        let mut compressed_bytes = Vec::new();
        a.serialize_compressed(&mut compressed_bytes)?;

        let a_compressed =
            <Bls12_381 as Pairing>::G1::deserialize_compressed(&compressed_bytes[..])?;
        assert_eq!(a, a_compressed);

        Ok(())
    }
}
