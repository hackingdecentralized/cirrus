use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use super::{prelude::DistributedError, MasterProverChannel, WorkerProverChannel};
use std::net::TcpStream;
use std::io::{Read, Write};

// Structs for the Master and Worker channels using synchronous socket communication
pub struct MasterProverChannelSocket {
    log_num_workers: usize,
    worker_sockets: Vec<TcpStream>,
}

pub struct WorkerProverChannelSocket {
    worker_id: usize,
    socket: TcpStream,
}

// Implement MasterProverChannel for MasterProverChannelSocket with synchronous send and recv
impl MasterProverChannel for MasterProverChannelSocket {
    fn send(&mut self, msg: &impl CanonicalSerialize) -> Result<(), DistributedError> {
        let mut serialized_msg = Vec::new();
        msg.serialize_compressed(&mut serialized_msg).map_err(DistributedError::from)?;

        for worker_socket in &self.worker_sockets {
            let mut socket = worker_socket.try_clone().map_err(|_| DistributedError::MasterSendError)?;

            // Send message length first
            let msg_len = (serialized_msg.len() as u64).to_le_bytes();
            socket.write_all(&msg_len).map_err(|_| DistributedError::MasterSendError)?;

            // Send the actual message
            socket.write_all(&serialized_msg).map_err(|_| DistributedError::MasterSendError)?;
        }

        Ok(())
    }

    fn send_all<T: CanonicalSerialize + Send>(&mut self, msg: Vec<T>) -> Result<(), DistributedError> {
        for (i, worker_socket) in self.worker_sockets.iter().enumerate() {
            let mut serialized_msg = Vec::new();
            msg[i].serialize_compressed(&mut serialized_msg).map_err(DistributedError::from)?;

            let mut socket = worker_socket.try_clone().map_err(|_| DistributedError::MasterSendError)?;

            let msg_len = (serialized_msg.len() as u64).to_le_bytes();
            socket.write_all(&msg_len).map_err(|_| DistributedError::MasterSendError)?;
            socket.write_all(&serialized_msg).map_err(|_| DistributedError::MasterSendError)?;
        }

        Ok(())
    }

    fn recv<T: CanonicalDeserialize + Send>(&mut self) -> Result<Vec<T>, DistributedError> {
        let mut results = Vec::new();
        for worker_socket in &self.worker_sockets {
            let mut socket = worker_socket.try_clone().map_err(|_| DistributedError::MasterRecvError)?;

            // Receive message length first
            let mut len_buf = [0u8; 8];
            socket.read_exact(&mut len_buf).map_err(|_| DistributedError::MasterRecvError)?;
            let msg_len = u64::from_le_bytes(len_buf) as usize;

            // Receive the actual message
            let mut buffer = vec![0; msg_len];
            socket.read_exact(&mut buffer).map_err(|_| DistributedError::MasterRecvError)?;

            let msg = T::deserialize_compressed(&buffer[..]).map_err(DistributedError::from)?;
            results.push(msg);
        }

        Ok(results)
    }

    fn log_num_workers(&self) -> usize {
        self.log_num_workers
    }
}

// Implement WorkerProverChannel for WorkerProverChannelSocket with synchronous send and recv
impl WorkerProverChannel for WorkerProverChannelSocket {
    fn send(&mut self, msg: &(impl CanonicalSerialize + Send)) -> Result<(), DistributedError> {
        let mut serialized_msg = Vec::new();
        msg.serialize_compressed(&mut serialized_msg).map_err(DistributedError::from)?;

        let msg_len = (serialized_msg.len() as u64).to_le_bytes();
        self.socket.write_all(&msg_len).map_err(|_| DistributedError::WorkerSendError)?;
        self.socket.write_all(&serialized_msg).map_err(|_| DistributedError::WorkerSendError)?;

        Ok(())
    }

    fn recv<T: CanonicalDeserialize>(&mut self) -> Result<T, DistributedError> {
        let mut len_buf = [0u8; 8];
        self.socket.read_exact(&mut len_buf).map_err(|_| DistributedError::WorkerRecvError)?;
        let msg_len = u64::from_le_bytes(len_buf) as usize;

        let mut buffer = vec![0; msg_len];
        self.socket.read_exact(&mut buffer).map_err(|_| DistributedError::WorkerRecvError)?;

        let msg = T::deserialize_compressed(&buffer[..]).map_err(DistributedError::from)?;
        Ok(msg)
    }

    fn worker_id(&self) -> usize {
        self.worker_id
    }
}

// Function to initialize Master and Worker channels over synchronous sockets
pub fn new_master_worker_socket_channels(
    log_num_workers: usize,
    worker_addrs: Vec<&str>
) -> (MasterProverChannelSocket, Vec<WorkerProverChannelSocket>) {
    let mut worker_sockets = Vec::new();
    let mut worker_channels = Vec::new();

    for (worker_id, addr) in worker_addrs.into_iter().enumerate() {
        let socket = TcpStream::connect(addr).map_err(|_| DistributedError::ChCreateError).expect("Failed to create socket");
        worker_sockets.push(socket.try_clone().map_err(|_| DistributedError::ChCreateError).expect("Failed to clone socket"));
        worker_channels.push(WorkerProverChannelSocket {
            worker_id,
            socket
        });
    }

    let master_channel = MasterProverChannelSocket {
        log_num_workers,
        worker_sockets,
    };

    (master_channel, worker_channels)
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_std::UniformRand;
    use std::thread;
    use std::net::TcpListener;

    #[test]
    fn test_master_worker_socket_channels() {
        // Define the number of workers
        let num_workers = 2;
        let mut worker_handles = Vec::new();

        // Start multiple workers in separate threads, each binding to a unique port
        for worker_id in 0..num_workers {
            let port = 7878 + worker_id; // Use different ports for each worker
            let handle = thread::spawn(move || {
                let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).unwrap();
                let (mut socket, _) = listener.accept().unwrap();

                // Worker receives a message
                let mut len_buf = [0u8; 8];
                socket.read_exact(&mut len_buf).unwrap();
                let msg_len = u64::from_le_bytes(len_buf) as usize;

                let mut buffer = vec![0; msg_len];
                socket.read_exact(&mut buffer).unwrap();

                // Echo back the received message
                socket.write_all(&len_buf).unwrap();
                socket.write_all(&buffer).unwrap();
            });

            worker_handles.push(handle);
        }

        // Initialize master and worker channels
        let worker_addrs: Vec<String> = (0..num_workers)
            .map(|i| format!("127.0.0.1:{}", 7878 + i))
            .collect();

        let (mut master_channel, _worker_channels) = new_master_worker_socket_channels(num_workers, worker_addrs.iter().map(|s| s.as_str()).collect());

        // Master sends a message to all workers
        let mut rng = ark_std::test_rng();
        let a = <Bls12_381 as Pairing>::G1::rand(&mut rng);

        master_channel.send(&a).unwrap();

        // Master receives the echoed messages back from all workers
        let received: Vec<<Bls12_381 as Pairing>::G1> = master_channel.recv().unwrap();

        for i in 0..num_workers {
            assert_eq!(received[i], a);
        }

        // Join the worker threads to ensure all threads finish execution
        for handle in worker_handles {
            handle.join().unwrap();
        }
    }
}