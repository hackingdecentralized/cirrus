use std::sync::mpsc::{channel, Receiver, Sender};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator
};

use super::{prelude::DistributedError, MasterProverChannel, WorkerProverChannel};

pub struct MasterProverChannelThread {
    log_num_workers: usize,
    send_channel: Vec<Sender<Vec<u8>>>,
    recv_channel: Vec<Receiver<Vec<u8>>>,
}

pub struct WorkerProverChannelThread {
    worker_id: usize,
    send_channel: Sender<Vec<u8>>,
    recv_channel: Receiver<Vec<u8>>,
}

impl MasterProverChannelThread {
    pub fn new(
        log_num_workers: usize,
        send_channel: Vec<Sender<Vec<u8>>>,
        recv_channel: Vec<Receiver<Vec<u8>>>,
    ) -> Self {
        assert_eq!(send_channel.len(), 1 << log_num_workers);
        assert_eq!(recv_channel.len(), 1 << log_num_workers);

        Self {
            log_num_workers,
            send_channel,
            recv_channel,
        }
    }
}

impl WorkerProverChannelThread {
    pub fn new(
        worker_id: usize,
        send_channel: Sender<Vec<u8>>,
        recv_channel: Receiver<Vec<u8>>,
    ) -> Self {
        Self {
            worker_id,
            send_channel,
            recv_channel,
        }
    }
}

impl MasterProverChannel for MasterProverChannelThread {
    fn send_uniform(&mut self, msg: &impl CanonicalSerialize) -> Result<(), DistributedError> {
        let mut serialized_msg = Vec::new();
        msg.serialize_compressed(&mut serialized_msg)
            .map_err(DistributedError::from)?;

        #[cfg(feature = "parallel")]
        self.send_channel
            .par_iter()
            .map(|channel| {
                channel
                    .send(serialized_msg.clone())
                    .map_err(|_| DistributedError::MasterSendError)
            })
            .collect::<Result<Vec<_>, DistributedError>>()?;

        #[cfg(not(feature = "parallel"))]
        self.send_channel
            .iter()
            .map(|channel| {
                channel
                    .send(serialized_msg.clone())
                    .map_err(|_| DistributedError::MasterSendError)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }

    fn send_different<T: CanonicalSerialize + Send>(
        &mut self,
        msg: Vec<T>,
    ) -> Result<(), DistributedError> {
        #[cfg(feature = "parallel")]
        self.send_channel
            .par_iter()
            .zip(msg.into_par_iter())
            .map(|(channel, msg)| {
                let mut serialized_msg = Vec::new();
                msg.serialize_compressed(&mut serialized_msg)
                    .map_err(DistributedError::from)?;
                channel
                    .send(serialized_msg)
                    .map_err(|_| DistributedError::MasterSendError)
            })
            .collect::<Result<Vec<_>, _>>()?;

        #[cfg(not(feature = "parallel"))]
        self.send_channel
            .iter()
            .zip(msg.into_iter())
            .map(|(channel, msg)| {
                let mut serialized_msg = Vec::new();
                msg.serialize_compressed(&mut serialized_msg)
                    .map_err(DistributedError::from)?;
                channel
                    .send(serialized_msg)
                    .map_err(|_| DistributedError::MasterSendError)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }

    /// TODO: Can you make it parallel?
    fn recv<T: CanonicalDeserialize + Send>(&mut self) -> Result<Vec<T>, DistributedError> {
        #[cfg(feature = "parallel")]
        return self.recv_channel.par_iter_mut().map(| channel | {
            let received_msg = channel.recv().map_err(|_| DistributedError::MasterRecvError)?;
            T::deserialize_compressed(&received_msg[..])
                .map_err(DistributedError::from)
        }).collect();

        #[cfg(not(feature = "parallel"))]
        return self
            .recv_channel
            .iter()
            .map(|channel| {
                let received_msg = channel
                    .recv()
                    .map_err(|_| DistributedError::MasterRecvError)?;
                T::deserialize_compressed(&received_msg[..]).map_err(DistributedError::from)
            })
            .collect();
    }

    fn log_num_workers(&self) -> usize {
        self.log_num_workers
    }
}

impl WorkerProverChannel for WorkerProverChannelThread {
    fn send(&mut self, msg: &(impl CanonicalSerialize + Send)) -> Result<(), DistributedError> {
        let mut serialized_msg = Vec::new();
        msg.serialize_compressed(&mut serialized_msg)
            .map_err(DistributedError::from)?;
        self.send_channel
            .send(serialized_msg)
            .map_err(|_| DistributedError::WorkerSendError)
    }

    fn recv<T: CanonicalDeserialize>(&mut self) -> Result<T, DistributedError> {
        let received_msg = self
            .recv_channel
            .recv()
            .map_err(|_| DistributedError::WorkerRecvError)?;
        T::deserialize_compressed(&received_msg[..]).map_err(DistributedError::from)
    }

    fn worker_id(&self) -> usize {
        self.worker_id
    }
}

pub fn new_master_worker_thread_channels(
    log_num_workers: usize,
) -> (MasterProverChannelThread, Vec<WorkerProverChannelThread>) {
    let num_workers = 1 << log_num_workers;
    let (master_send, worker_recv): (Vec<_>, Vec<_>) = (0..num_workers).map(|_| channel()).unzip();

    let (worker_send, master_recv): (Vec<_>, Vec<_>) = (0..num_workers).map(|_| channel()).unzip();

    let master_channel = MasterProverChannelThread::new(log_num_workers, master_send, master_recv);
    let worker_channels = worker_send
        .into_iter()
        .zip(worker_recv.into_iter())
        .enumerate()
        .map(|(worker_id, (send_uniform, recv))| {
            WorkerProverChannelThread::new(worker_id, send_uniform, recv)
        })
        .collect();

    (master_channel, worker_channels)
}

#[cfg(test)]
mod test {
    use std::thread::spawn;

    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    type ScalarField = <Bls12_381 as Pairing>::ScalarField;

    use super::*;

    #[test]
    fn test_master_worker_thread_channels() {
        let log_num_workers = 2;
        let (mut master_channel, mut worker_channels) =
            new_master_worker_thread_channels(log_num_workers);

        let master_send = vec![1, 2, 3];
        master_channel.send_uniform(&master_send).unwrap();
        let received_msgs: Vec<Vec<u8>> = worker_channels
            .iter_mut()
            .map(|worker_channel| worker_channel.recv::<Vec<u8>>().unwrap())
            .collect();
        assert_eq!(received_msgs, vec![master_send.clone(); 4]);
    }

    #[test]
    fn test_multiple_thread() {
        let log_num_workers = 2;
        let (mut master_channel, worker_channels) =
            new_master_worker_thread_channels(log_num_workers);
        let master = spawn(move || {
            let master_send = vec![1u8, 2, 3];
            master_channel.send_uniform(&master_send).unwrap();
            let receive: Vec<u8> = master_channel.recv().unwrap();
            println!("{:?}", receive);
            assert_eq!(
                receive,
                (0..1 << log_num_workers).collect::<Vec<u8>>(),
                "Received message is not equal to the sent message"
            );

            let receive: Vec<ScalarField> = master_channel.recv().unwrap();
            println!("{:?}", receive);
            assert_eq!(
                receive,
                vec![1u128; 1 << log_num_workers]
                    .into_iter()
                    .map(|x| ScalarField::from(x))
                    .collect::<Vec<_>>(),
                "Received message is not equal to the sent message"
            );
        });

        let workers: Vec<_> = worker_channels
            .into_iter()
            .rev()
            .map(|mut worker_channel| {
                spawn(move || {
                    let received_msg: Vec<u8> = worker_channel.recv().unwrap();
                    assert_eq!(
                        received_msg,
                        vec![1u8, 2, 3],
                        "Received message is not equal to the sent message"
                    );

                    worker_channel
                        .send(&(worker_channel.worker_id() as u8))
                        .unwrap();

                    worker_channel.send(&ScalarField::from(1u128)).unwrap();
                })
            })
            .collect();

        master.join().expect("Master thread panicked");
        workers
            .into_iter()
            .map(|worker| worker.join())
            .collect::<Result<Vec<_>, _>>()
            .expect("worker thread panicked");
    }
}
