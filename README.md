# Cirrus: Performant and Accountable Distributed SNARK with Linear Prover Time

## Run master prover and worker prover

To run, please first change `scripts/run.sh` to executable

```bash
> chmod +x scripts/run.sh
```

To run local experiments, first run the setup

```bash
> scripts/run.sh setup --log_num_workers $log_num_workers --log_num_vars $log_num_vars
```

### Run locally

In one terminal, run the master

```bash
> scripts/run.sh run_master --log_num_workers $log_num_workers --log_num_vars $log_num_vars
```

You can find the log in the folder `out/vanilla-$log_num_vars-$log_num_workers`. After the log shows `Master listening on ...`, run the workers in another terminal

```bash
> scripts/run.sh run_multi_worker --log_num_workers $log_num_workers --log_num_vars $log_num_vars
```

### Run on multiple machines

Make sure you have sent the keys to the corresponding master or worker.

On the master machine, run

```bash
> scripts/run.sh run_master --log_num_workers $log_num_workers --log_num_vars $log_num_vars --master_addr $master_addr:$master_listen_port
```

On each worker node, run

```bash
> scripts/run.sh run_single_worker --log_num_workers $log_num_workers --log_num_vars $log_num_vars --master_addr $master_addr:$master_listen_port
```

### Analyze results

To analyze the final log of the master, run

```bash
> scripts/run.sh analyze --log_num_workers $log_num_workers --log_num_vars $log_num_vars --analyze_target master
```

## Disclaimer

**DISCLAIMER:** This software is provided "as is" and its security has not been externally audited. Use at your own risk.

## Development environment setup

Our reference setup is on Ubuntu Noble 24.04 AMD64

### Install RUST

```bash
> curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
> source ~/.cargo/env
```

## Install Python (Optional)

We recommand MiniConda:

```bash
> mkdir -p ~/miniconda3
> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
> bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
> rm ~/miniconda3/miniconda.sh
> source ~/miniconda3/bin/activate
> conda init --all
```

### Tests

```
> cargo test --release --all
```

### HyperPlonk Benchmarks

HyperPlonk: A linear-time FFT-free SNARK proof system (https://eprint.iacr.org/2022/1355.pdf).

To obtain benchmarks of HyperPlonk, run the script file `scripts/run_benchmarks.sh`. 
We refer to Table 5 and Table 6 in https://eprint.iacr.org/2022/1355.pdf for an example benchmark.
