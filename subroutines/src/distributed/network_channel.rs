use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use super::{prelude::DistributedError, MasterProverChannel, WorkerProverChannel};
use std::net::{TcpStream, TcpListener};
use std::io::{Read, Write};

// Structs for the Master and Worker channels using persistent socket connections
pub struct MasterProverChannelSocket {
    log_num_workers: usize,
    worker_sockets: Vec<TcpStream>, // Stores accepted worker connections
}

pub struct WorkerProverChannelSocket {
    worker_id: usize,
    socket: TcpStream,  // Connection to the master
}

// Implement MasterProverChannel for MasterProverChannelSocket
impl MasterProverChannel for MasterProverChannelSocket {
    fn send_uniform(&mut self, msg: &impl CanonicalSerialize) -> Result<(), DistributedError> {
        let mut serialized_msg = Vec::new();
        msg.serialize_compressed(&mut serialized_msg).map_err(DistributedError::from)?;

        for worker_socket in &self.worker_sockets {
            let mut socket = worker_socket.try_clone().map_err(|_| DistributedError::MasterSendError)?;

            let msg_len = (serialized_msg.len() as u64).to_le_bytes();
            socket.write_all(&msg_len).map_err(|_| DistributedError::MasterSendError)?;
            socket.write_all(&serialized_msg).map_err(|_| DistributedError::MasterSendError)?;
        }

        Ok(())
    }

    fn send_different<T: CanonicalSerialize + Send>(&mut self, msgs: Vec<T>) -> Result<(), DistributedError> {
        if msgs.len() != self.worker_sockets.len() {
            return Err(DistributedError::MasterSendError);
        }

        for (i, worker_socket) in self.worker_sockets.iter().enumerate() {
            let mut serialized_msg = Vec::new();
            msgs[i].serialize_compressed(&mut serialized_msg).map_err(DistributedError::from)?;

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

            let mut len_buf = [0u8; 8];
            socket.read_exact(&mut len_buf).map_err(|_| DistributedError::MasterRecvError)?;
            let msg_len = u64::from_le_bytes(len_buf) as usize;

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

// Implement WorkerProverChannel for WorkerProverChannelSocket with send and recv
impl WorkerProverChannel for WorkerProverChannelSocket {
    fn send(&mut self, msg: &(impl CanonicalSerialize + Send)) -> Result<(), DistributedError> {
        let mut serialized_msg = Vec::new();
        msg.serialize_compressed(&mut serialized_msg).map_err(DistributedError::from)?;

        let msg_len = (serialized_msg.len() as u64).to_le_bytes();

        let mut socket = self.socket.try_clone().map_err(|_| DistributedError::WorkerSendError)?;

        // Send message length to the master's socket
        socket.write_all(&msg_len).map_err(|_| DistributedError::WorkerSendError)?;

        // Send the actual message to the master's socket
        socket.write_all(&serialized_msg).map_err(|_| DistributedError::WorkerSendError)?;

        Ok(())
    }

    fn recv<T: CanonicalDeserialize>(&mut self) -> Result<T, DistributedError> {
        let mut socket = self.socket.try_clone().map_err(|_| DistributedError::WorkerRecvError)?;

        // Receive message length from the master
        let mut len_buf = [0u8; 8];
        socket.read_exact(&mut len_buf).map_err(|_| DistributedError::WorkerRecvError)?;
        let msg_len = u64::from_le_bytes(len_buf) as usize;

        // Receive the actual message from the master
        let mut buffer = vec![0; msg_len];
        socket.read_exact(&mut buffer).map_err(|_| DistributedError::WorkerRecvError)?;

        let msg = T::deserialize_compressed(&buffer[..]).map_err(DistributedError::from)?;
        Ok(msg)
    }

    fn worker_id(&self) -> usize {
        self.worker_id
    }
}

// Function to initialize Master and Worker channels with Master as a listening server
pub fn new_master_worker_socket_channels(
    log_num_workers: usize,
    master_addr: &str,
) -> (MasterProverChannelSocket, Vec<WorkerProverChannelSocket>) {
    // Master starts listening on the specified address
    let listener = TcpListener::bind(master_addr).expect("Failed to bind master listener");

    let num_workers = 1<<log_num_workers;

    let master_socket_addr = listener.local_addr().expect("Failed to get local address");

    // Accept incoming worker connections up to log_num_workers
    // for worker_id in 0..(1<<log_num_workers) {
    //     let (worker_socket, _addr) = listener.accept().expect("Failed to accept worker connection");
    //     worker_channels.push(WorkerProverChannelSocket {
    //         worker_id,
    //         socket: worker_socket,
    //     });
    // }

    // let master_channel = MasterProverChannelSocket {
    //     log_num_workers,
    //     worker_sockets: worker_channels.iter().map(|wc| wc.socket.try_clone().unwrap()).collect(),
    // };
    let worker_channels: Vec<WorkerProverChannelSocket> = (0..num_workers)
            .map(|worker_id| {
                let socket = TcpStream::connect(master_socket_addr).expect("Failed to connect worker to master");
                WorkerProverChannelSocket {
                    worker_id,
                    socket,
                }
            })
            .collect();

        // Accept incoming worker connections in master
        let mut worker_sockets = Vec::new();
        for _ in 0..(num_workers) {
            let (socket, _addr) = listener.accept().expect("Failed to accept worker connection");
            worker_sockets.push(socket);
        }

        let master_channel = MasterProverChannelSocket {
            log_num_workers: log_num_workers,
            worker_sockets,
        };

    (master_channel, worker_channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::net::{TcpListener, TcpStream};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use std::sync::{Arc, Mutex};
    // use std::time::Duration;

    // Mock message struct for serialization/deserialization
    #[derive(CanonicalSerialize, CanonicalDeserialize, PartialEq, Debug, Clone)]
    struct TestMessage {
        data: u64,
    }

    // Helper function to start a listener for the master channel and accept worker connections
    fn setup_master_and_worker_channels(num_workers: usize) -> (MasterProverChannelSocket, Vec<WorkerProverChannelSocket>) {
        let master_addr = "127.0.0.1:0"; // Bind to an available port
        let listener = TcpListener::bind(master_addr).expect("Failed to bind listener");
        let master_socket_addr = listener.local_addr().expect("Failed to get local address");

        let worker_channels: Vec<WorkerProverChannelSocket> = (0..num_workers)
            .map(|worker_id| {
                let socket = TcpStream::connect(master_socket_addr).expect("Failed to connect worker to master");
                WorkerProverChannelSocket {
                    worker_id,
                    socket,
                }
            })
            .collect();

        // Accept incoming worker connections in master
        let mut worker_sockets = Vec::new();
        for _ in 0..num_workers {
            let (socket, _addr) = listener.accept().expect("Failed to accept worker connection");
            worker_sockets.push(socket);
        }

        let master_channel = MasterProverChannelSocket {
            log_num_workers: num_workers,
            worker_sockets,
        };

        (master_channel, worker_channels)
    }

    #[test]
    fn test_send_uniform() {
        let (mut master_channel, worker_channels) = new_master_worker_socket_channels(1, "127.0.0.1:0");

        let msg = TestMessage { data: 42 };
        let worker_handles: Vec<_> = worker_channels
            .into_iter()
            .map(|mut worker| {
                let msg_clone = msg.clone();
                thread::spawn(move || {
                    let received_msg: TestMessage = worker.recv().expect("Failed to receive message");
                    assert_eq!(received_msg, msg_clone);
                })
            })
            .collect();

        // Master sends a uniform message to all workers
        master_channel.send_uniform(&msg).expect("Failed to send uniform message");

        for handle in worker_handles {
            handle.join().expect("Worker thread panicked");
        }
    }

    #[test]
    fn test_send_different() {
        let (mut master_channel, worker_channels) = new_master_worker_socket_channels(1, "127.0.0.1:0");

        let msgs = vec![
            TestMessage { data: 42 },
            TestMessage { data: 99 },
        ];

        let worker_handles: Vec<_> = worker_channels
            .into_iter()
            .enumerate()
            .map(|(i, mut worker)| {
                let expected_msg = msgs[i].clone();
                thread::spawn(move || {
                    let received_msg: TestMessage = worker.recv().expect("Failed to receive message");
                    assert_eq!(received_msg, expected_msg);
                })
            })
            .collect();

        // Master sends different messages to each worker
        master_channel.send_different(msgs).expect("Failed to send different messages");

        for handle in worker_handles {
            handle.join().expect("Worker thread panicked");
        }
    }

    #[test]
    fn test_worker_send_to_master() {
        let (mut master_channel, worker_channels) = new_master_worker_socket_channels(1, "127.0.0.1:0");

        let msgs = vec![
            TestMessage { data: 42 },
            TestMessage { data: 99 },
        ];

        // Workers send messages to the master
        let worker_handles: Vec<_> = worker_channels
            .into_iter()
            .enumerate()
            .map(|(i, mut worker)| {
                let msg = msgs[i].clone();
                thread::spawn(move || {
                    worker.send(&msg).expect("Failed to send message");
                })
            })
            .collect();

        // Ensure workers complete sending
        for handle in worker_handles {
            handle.join().expect("Worker thread panicked");
        }

        // Master receives messages from workers
        let received_msgs: Vec<TestMessage> = master_channel.recv().expect("Failed to receive messages from workers");

        assert_eq!(received_msgs, msgs);
    }

    #[test]
    fn test_multithreaded_worker_communication() {
        let (mut master_channel, worker_channels) = new_master_worker_socket_channels(2, "127.0.0.1:0");

        let worker_channels = Arc::new(Mutex::new(worker_channels));
        let master_channel = Arc::new(Mutex::new(master_channel));
        let message = TestMessage { data: 123 };

        // Master sends messages in a separate thread
        let master_send_thread = {
            let master_channel = Arc::clone(&master_channel);
            let message = message.clone();
            thread::spawn(move || {
                let mut master_channel = master_channel.lock().unwrap();
                master_channel.send_uniform(&message).expect("Failed to send message to workers");
            })
        };

        // Workers each send a message to the master in parallel threads
        let worker_send_threads: Vec<_> = (0..4)
            .map(|worker_id| {
                let worker_channels = Arc::clone(&worker_channels);
                let message = TestMessage { data: worker_id as u64 + 100 };

                thread::spawn(move || {
                    let mut workers = worker_channels.lock().unwrap();
                    let mut worker = &mut workers[worker_id];
                    worker.send(&message).expect("Failed to send message to master");
                })
            })
            .collect();

        // Worker threads receive messages from the master in parallel
        let worker_receive_threads: Vec<_> = (0..4)
            .map(|worker_id| {
                let worker_channels = Arc::clone(&worker_channels);
                let expected_message = message.clone();

                thread::spawn(move || {
                    let mut workers = worker_channels.lock().unwrap();
                    let mut worker = &mut workers[worker_id];
                    let received_msg: TestMessage = worker.recv().expect("Failed to receive message from master");
                    assert_eq!(received_msg, expected_message);
                })
            })
            .collect();

        // Join all threads
        master_send_thread.join().expect("Master send thread panicked");
        for handle in worker_send_threads {
            handle.join().expect("Worker send thread panicked");
        }
        for handle in worker_receive_threads {
            handle.join().expect("Worker receive thread panicked");
        }

        // Master receives messages from workers
        let received_msgs: Vec<TestMessage> = master_channel.lock().unwrap().recv().expect("Failed to receive messages from workers");
        let expected_msgs: Vec<TestMessage> = (0..4).map(|worker_id| TestMessage { data: worker_id as u64 + 100 }).collect();
        assert_eq!(received_msgs, expected_msgs);
    }
}
