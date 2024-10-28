use super::{prelude::DistributedError, MasterProverChannel, WorkerProverChannel};
use super::{thread_channel, network_channel};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub enum MasterProverChannelEnum {
    Thread(thread_channel::MasterProverChannelThread),
    Socket(network_channel::MasterProverChannelSocket),
}

// Enum to encapsulate both thread and socket implementations for WorkerProverChannel
pub enum WorkerProverChannelEnum {
    Thread(thread_channel::WorkerProverChannelThread),
    Socket(network_channel::WorkerProverChannelSocket),
}

// Implement MasterProverChannel for the enum
impl MasterProverChannel for MasterProverChannelEnum {
    fn send_uniform(&mut self, msg: &impl CanonicalSerialize) -> Result<(), DistributedError> {
        match self {
            MasterProverChannelEnum::Thread(channel) => channel.send_uniform(msg),
            MasterProverChannelEnum::Socket(channel) => channel.send_uniform(msg),
        }
    }

    fn send_different<T: CanonicalSerialize + Send>(&mut self, msg: Vec<T>) -> Result<(), DistributedError> {
        match self {
            MasterProverChannelEnum::Thread(channel) => channel.send_different(msg),
            MasterProverChannelEnum::Socket(channel) => channel.send_different(msg),
        }
    }

    fn recv<T: CanonicalDeserialize + Send>(&mut self) -> Result<Vec<T>, DistributedError> {
        match self {
            MasterProverChannelEnum::Thread(channel) => channel.recv(),
            MasterProverChannelEnum::Socket(channel) => channel.recv(),
        }
    }

    fn log_num_workers(&self) -> usize {
        match self {
            MasterProverChannelEnum::Thread(channel) => channel.log_num_workers(),
            MasterProverChannelEnum::Socket(channel) => channel.log_num_workers(),
        }
    }
}

// Implement WorkerProverChannel for the enum
impl WorkerProverChannel for WorkerProverChannelEnum {
    fn send(&mut self, msg: &(impl CanonicalSerialize + Send)) -> Result<(), DistributedError> {
        match self {
            WorkerProverChannelEnum::Thread(channel) => channel.send(msg),
            WorkerProverChannelEnum::Socket(channel) => channel.send(msg),
        }
    }

    fn recv<T: CanonicalDeserialize>(&mut self) -> Result<T, DistributedError> {
        match self {
            WorkerProverChannelEnum::Thread(channel) => channel.recv(),
            WorkerProverChannelEnum::Socket(channel) => channel.recv(),
        }
    }

    fn worker_id(&self) -> usize {
        match self {
            WorkerProverChannelEnum::Thread(channel) => channel.worker_id(),
            WorkerProverChannelEnum::Socket(channel) => channel.worker_id(),
        }
    }
}

// Factory function to initialize either thread or socket channels
pub fn new_master_worker_channels(
    use_sockets: bool,
    log_num_workers: usize,
    master_addr: &str
) -> (MasterProverChannelEnum, Vec<WorkerProverChannelEnum>) {
    if use_sockets {
        let (master_socket, worker_sockets) = network_channel::new_master_worker_socket_channels(log_num_workers, master_addr);
        let master = MasterProverChannelEnum::Socket(master_socket);
        let workers = worker_sockets
            .into_iter()
            .map(WorkerProverChannelEnum::Socket)
            .collect();

        (master, workers)
    } else {
        let (master_thread, worker_threads) = thread_channel::new_master_worker_thread_channels(log_num_workers);
        let master = MasterProverChannelEnum::Thread(master_thread);
        let workers = worker_threads
            .into_iter()
            .map(WorkerProverChannelEnum::Thread)
            .collect();

        (master, workers)
    }
}