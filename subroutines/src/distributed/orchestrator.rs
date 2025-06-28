use std::sync::mpsc::{channel, Receiver, Sender};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_ff::PrimeField;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use super::{prelude::DistributedError, MasterProverChannel, WorkerProverChannel};

use std::path::PathBuf;
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Orchestrator manages the file paths and loading/storing 
/// of worker polynomials in a single-machine environment.
pub struct Orchestrator {
    /// log_num_workers indicates we have 2^log_num_workers total workers.
    pub log_num_workers: usize,
    /// A list of file paths (one per worker) pointing to each worker's data on disk.
    /// Typically we expect `worker_files.len() == 2^log_num_workers`.
    pub worker_files: Vec<PathBuf>,

    // Optionally, you can store any other config or ephemeral paths 
    // here. For example:
    // pub temp_dir: PathBuf,

    // If desired, you can include something like `PhantomData<F>` to
    // avoid Rust complaining about F not being used in the struct layout.
    // But if you store polynomials of type `VirtualPolynomial<F>` in memory,
    // that’s typically enough.
}

impl Orchestrator {
    /// Construct a new Orchestrator with the given log_num_workers and file paths.
    /// 
    /// # Arguments
    /// - `log_num_workers`: The exponent for how many total workers there are.
    /// - `worker_files`: A vector of file paths, one for each worker’s polynomial.
    ///
    /// # Panics
    /// If `worker_files.len() != 2^log_num_workers`.
    pub fn new(log_num_workers: usize, worker_files: Vec<PathBuf>) -> Self {
        let expected_len = 1 << log_num_workers;
        assert_eq!(
            worker_files.len(),
            expected_len,
            "Expected {} files for {} workers (2^log_num_workers). Got {} instead.",
            expected_len,
            log_num_workers,
            worker_files.len(),
        );

        Orchestrator {
            log_num_workers,
            worker_files,
        }
    }

    /// Load a worker polynomial from disk given a worker ID.
    ///
    /// # Arguments
    /// - `worker_id`: The ID of the worker (0..2^log_num_workers).
    ///
    /// # Returns
    /// A `VirtualPolynomial<F>` on success, or `PolyIOPErrors` if something goes wrong (e.g. I/O failure).
    pub fn load_worker_polynomial(
        &self,
        worker_id: usize,
    ) -> Result<VirtualPolynomial<F>, PolyIOPErrors> {
        // 1) Validate worker_id
        if worker_id >= (1 << self.log_num_workers) {
            return Err(PolyIOPErrors::InvalidWorkerNumber);
        }

        // 2) Retrieve path and open file
        let path = &self.worker_files[worker_id];
        let file = File::open(path)
            .map_err(|e| PolyIOPErrors::InvalidParameters(format!("I/O error: {e}")))?;

        // 3) Read polynomial from file. You can use your own serialization
        // method: bincode, serde, custom, etc. We use pseudo-code here:
        let mut reader = BufReader::new(file);
        let poly: VirtualPolynomial<F> = bincode::deserialize_from(&mut reader)
            .map_err(|e| PolyIOPErrors::InvalidParameters(format!("Deserialization error: {e}")))?;

        Ok(poly)
    }

    /// Store a worker polynomial to disk at the worker's file path.
    /// 
    /// You might call this if you need to *update* the polynomial on disk after
    /// partial modifications. If you only read from disk, you might not need this.
    pub fn store_worker_polynomial(
        &self,
        worker_id: usize,
        worker_poly: &VirtualPolynomial<F>,
    ) -> Result<(), PolyIOPErrors> {
        // Validate
        if worker_id >= (1 << self.log_num_workers) {
            return Err(PolyIOPErrors::InvalidWorkerNumber);
        }
        let path = &self.worker_files[worker_id];

        // Open file for writing
        let file = File::create(path)
            .map_err(|e| PolyIOPErrors::InvalidParameters(format!("I/O error: {e}")))?;
        let mut writer = BufWriter::new(file);

        // Serialize and write the polynomial
        bincode::serialize_into(&mut writer, &worker_poly)
            .map_err(|e| PolyIOPErrors::InvalidParameters(format!("Serialization error: {e}")))?;

        Ok(())
    }

    // Optionally, you can define a helper function to iterate over all workers 
    // and do something with them. For example, a higher-level “apply_round”
    // routine that loads each worker, updates state, and stores it back:
    //
    // pub fn apply_round(
    //    &self,
    //    current_challenge: Option<F>,
    // ) -> Result<Vec<IOPProverMessage<F>>, PolyIOPErrors> {
    //     let mut aggregator = vec![F::zero(); ???]; // aggregator for the current round
    // 
    //     for worker_id in 0..(1 << self.log_num_workers) {
    //         let mut worker_poly = self.load_worker_polynomial(worker_id)?;
    //         let msg = do_one_round(&mut worker_poly, current_challenge)?; 
    //         // aggregator += msg
    //         // you might store back if worker_poly changed
    //         // self.store_worker_polynomial(worker_id, &worker_poly)?;
    //     }
    //     ...
    // }
    //
    // But the exact structure depends on how your protocol is orchestrated.
}