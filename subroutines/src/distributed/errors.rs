use displaydoc::Display;

#[derive(Debug, Display)]
pub enum DistributedError {
    /// An error during (de)serialization: {0}
    SerializationErrors(ark_serialize::SerializationError),
    /// Master prover failed to send_uniform message
    MasterSendError,
    /// Master prover failed to receive message
    MasterRecvError,
    /// Worker prover failed to send_uniform message
    WorkerSendError,
    /// Worker prover failed to receive message
    WorkerRecvError,
    /// Channel creat error
    ChCreateError,
}

impl From<ark_serialize::SerializationError> for DistributedError {
    fn from(e: ark_serialize::SerializationError) -> Self {
        Self::SerializationErrors(e)
    }
}