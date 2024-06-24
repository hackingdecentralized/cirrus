use std::sync::mpsc::{channel, Receiver, Sender};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{prelude::DistributedError, MasterProverChannel, SlaveProverChannel};

pub struct MasterProverChannelThread {
    log_num_slaves: usize,
    send_channel: Vec<Sender<Vec<u8>>>,
    recv_channel: Vec<Receiver<Vec<u8>>>,
}

pub struct SlaveProverChannelThread {
    slave_id: usize,
    send_channel: Sender<Vec<u8>>,
    recv_channel: Receiver<Vec<u8>>,
}

impl MasterProverChannelThread {
    pub fn new(
        log_num_slaves: usize,
        send_channel: Vec<Sender<Vec<u8>>>,
        recv_channel: Vec<Receiver<Vec<u8>>>,
    ) -> Self {
        assert_eq!(send_channel.len(), 1 << log_num_slaves);
        assert_eq!(recv_channel.len(), 1 << log_num_slaves);

        Self {
            log_num_slaves,
            send_channel,
            recv_channel,
        }
    }
}

impl SlaveProverChannelThread {
    pub fn new(slave_id: usize,
        send_channel: Sender<Vec<u8>>,
        recv_channel: Receiver<Vec<u8>>
    ) -> Self {
        Self {
            slave_id,
            send_channel,
            recv_channel,
        }
    }
}

impl MasterProverChannel for MasterProverChannelThread {
    fn send(&self, msg: &impl CanonicalSerialize) -> Result<(), DistributedError> {
        let mut serialized_msg = Vec::new();
        msg.serialize_compressed(&mut serialized_msg)
            .map_err(DistributedError::from)?;

        #[cfg(feature = "parallel")]
        self.send_channel.par_iter().map(| channel | {
            channel.send(serialized_msg.clone())
                .map_err(|_| DistributedError::MasterSendError)
        }).collect::<Result<Vec<_>, DistributedError>>()?;

        #[cfg(not(feature = "parallel"))]
        self.send_channel.iter().map(| channel | {
            channel.send(serialized_msg.clone())
                .map_err(|_| DistributedError::MasterSendError)
        }).collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }

    /// TODO: Can you make it parallel?
    fn recv<T: CanonicalDeserialize + Send>(&self) -> Result<Vec<T>, DistributedError> {
        // #[cfg(feature = "parallel")]
        // return self.recv_channel.par_iter_mut().map(| channel | {
        //     let received_msg = channel.recv().map_err(|_| DistributedError::MasterRecvError)?;
        //     T::deserialize_compressed(&received_msg[..])
        //         .map_err(DistributedError::from)
        // }).collect();

        // #[cfg(not(feature = "parallel"))]
        return self.recv_channel.iter().map(| channel | {
            let received_msg = channel.recv().map_err(|_| DistributedError::MasterRecvError)?;
            T::deserialize_compressed(&received_msg[..])
                .map_err(DistributedError::from)
        }).collect();
    }

    fn log_num_slaves(&self) -> usize {
        self.log_num_slaves
    }
}

impl SlaveProverChannel for SlaveProverChannelThread {
    fn send(&self, msg: &(impl CanonicalSerialize + Send)) -> Result<(), DistributedError> {
        let mut serialized_msg = Vec::new();
        msg.serialize_compressed(&mut serialized_msg).map_err(DistributedError::from)?;
        self.send_channel.send(serialized_msg).map_err(|_| DistributedError::SlaveSendError)
    }

    fn recv<T: CanonicalDeserialize>(&self) -> Result<T, DistributedError> {
        let received_msg = self.recv_channel.recv()
            .map_err(|_| DistributedError::SlaveRecvError)?;
        T::deserialize_compressed(&received_msg[..])
            .map_err(DistributedError::from)
    }

    fn slave_id(&self) -> usize {
        self.slave_id
    }
}

pub fn new_master_slave_thread_channels(
    log_num_slaves: usize
) -> (MasterProverChannelThread, Vec<SlaveProverChannelThread>) {
    let num_slaves = 1 << log_num_slaves;
    let (master_send, slave_recv): (Vec<_>, Vec<_>) = (0..num_slaves)
        .map(|_| { channel() }).unzip();

    let (slave_send, master_recv): (Vec<_>, Vec<_>) = (0..num_slaves)
        .map(|_| { channel() }).unzip();

    let master_channel = MasterProverChannelThread::new(log_num_slaves, master_send, master_recv);
    let slave_channels = slave_send.into_iter().zip(slave_recv.into_iter())
        .enumerate()
        .map( | (slave_id, (send, recv)) | {
            SlaveProverChannelThread::new(slave_id, send, recv)
        }).collect();

    (master_channel, slave_channels)
}

#[cfg(test)]
mod test {
    use std::thread::spawn;

    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    type ScalarField = <Bls12_381 as Pairing>::ScalarField;

    use super::*;

    #[test]
    fn test_master_slave_thread_channels() {
        let log_num_slaves = 2;
        let (master_channel, slave_channels) = new_master_slave_thread_channels(log_num_slaves);

        let master_send = vec![1, 2, 3];
        master_channel.send(&master_send).unwrap();
        let received_msgs: Vec<Vec<u8>> = slave_channels.iter().map(| slave_channel | {
            slave_channel.recv::<Vec<u8>>().unwrap()
        }).collect();
        assert_eq!(received_msgs, vec![master_send.clone(); 4]);
    }

    #[test]
    fn test_multiple_thread() {
        let log_num_slaves = 2;
        let (master_channel, slave_channels) = new_master_slave_thread_channels(log_num_slaves);
        let master = spawn(move || {
            let master_send = vec![1u8, 2, 3];
            master_channel.send(&master_send).unwrap();
            let receive: Vec<u8> = master_channel.recv().unwrap();
            println!("{:?}", receive);
            assert_eq!(receive, (0..1<<log_num_slaves).collect::<Vec<u8>>(), "Received message is not equal to the sent message");

            let receive: Vec<ScalarField> = master_channel.recv().unwrap();
            println!("{:?}", receive);
            assert_eq!(receive, vec![1u128; 1<<log_num_slaves]
                .into_iter()
                .map(| x | ScalarField::from(x))
                .collect::<Vec<_>>(),
                "Received message is not equal to the sent message");
        });

        let slaves: Vec<_> = slave_channels.into_iter().rev().map(| slave_channel | {
            spawn(move || {
                let received_msg: Vec<u8> = slave_channel.recv().unwrap();
                assert_eq!(received_msg, vec![1u8, 2, 3], "Received message is not equal to the sent message");

                slave_channel.send(&(slave_channel.slave_id() as u8)).unwrap();

                slave_channel.send(&ScalarField::from(1u128)).unwrap();
            })
        }).collect();

        master.join().expect("Master thread panicked");
        slaves.into_iter()
            .map(| slave | slave.join())
            .collect::<Result<Vec<_>, _>>()
            .expect("Slave thread panicked");
    }
}