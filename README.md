# Cirrus: Performant and Accountable Distributed SNARK with Linear Prover Time

## Test master prover and worker prover

To run the master prover and worker prover, please follow the instructions in `scripts/run.sh`.

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
