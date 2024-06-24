use displaydoc::Display;

#[derive(Debug, Display)]
pub enum DistributedError {
    /// An error during (de)serialization: {0}
    SerializationErrors(ark_serialize::SerializationError),
    /// Master prover failed to send message
    MasterSendError,
    /// Master prover failed to receive message
    MasterRecvError,
    /// Slave prover failed to send message
    SlaveSendError,
    /// Slave prover failed to receive message
    SlaveRecvError,
}

impl From<ark_serialize::SerializationError> for DistributedError {
    fn from(e: ark_serialize::SerializationError) -> Self {
        Self::SerializationErrors(e)
    }
}