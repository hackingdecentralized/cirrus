# Cirrus: Performant and Accountable Distributed SNARK

> [!WARNING]
This is an academic research prototype meant to elucidate protocol details for proofs-of-concept and benchmarking. It has not been developed for production usage, nor has it been audited (i.e., it may contain security vulnerabilities). 

## Overview
Cirrus is an accountable, distributed PLONK‑style prover built on top of HyperPlonk. Cirrus distributes proving across multiple workers, provides a fast accountability protocol to identify misbehaving workers, and uses a **universal** setup (SRS) compatible with HyperPlonk.

This README aligns with the AE appendix and is sufficient to:
- build and run the artifact locally or on AWS,
- reproduce the main evaluation figures (end‑to‑end time, accountability time, coordinator time),
- collect logs/JSON needed to verify the claims.

---

## What’s in this artifact
- **Source**: Rust implementation extending HyperPlonk by **5,000+** LOC.
- **Binaries / scripts**:
  - `scripts/run.sh` — setup, run coordinator (“master”), run workers, analyze logs.
  - `ec2/setup.py` — install prerequisites on AWS instances.
  - `ec2/transfer_setup_param.py` — upload precomputed setup (SRS) to AWS instances.
  - `ec2/run_exp_single_thread.py` — drive end‑to‑end distributed runs (E1).
  - `bench_master_ha.sh`, `bench_master_no_ha.sh` — coordinator timing (E3).
- **Benchmarks**: random circuits (scaling) + application kernels used in the paper.
- **Outputs**: logs and JSON under `out/vanilla-$log_num_vars-$log_num_workers/`.

---

## System requirements

### Hardware
**Distributed evaluation (paper scale).**
- Up to **32× AWS `t3.2xlarge`** (8 vCPUs, 32 GB RAM, 50 GB disk) in the **same region/VPC**.
- **One worker per core** (8 workers per machine) is the best setting and used in all experiments.

**Local machine (orchestrator / file staging).**
- If you **generate** setup locally: **64 GB RAM** and **500 GB** free disk.
- If you **download** precomputed setup: **8 GB RAM** and **500 GB** free disk.
- Public IP on each AWS instance (for simplicity during AE); see security notes below.

### Software
**AWS instances**
- OS: **Ubuntu Server 24.04 LTS**
- **Rust** (stable) and **Python 3.12** (installed by `ec2/setup.py`)

**Local machine**
- **Python 3.12** (Conda recommended)
- **Paramiko** (SSH automation)
- **Rust** (only if generating setup locally)
- Linux x86_64 recommended

---

## Quick start

### A. Local single‑host (functional validation)
This path validates functionality and trends on small circuits (e.g., up to \(2^{20}\)–\(2^{22}\) gates).

1) **Make runner executable**

    ```bash
    chmod +x scripts/run.sh
    ```

2.	(Optional) Install Rust (only to generate setup locally)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

3.	Generate or download setup (see Setup parameters￼).
Example (generate for (2^{{log_num_vars}}) gates with (2^{{log_num_workers}}) workers):

```bash
scripts/run.sh setup --log_num_workers $log_num_workers --log_num_vars $log_num_vars
```

4.	To test the code locally

    - Terminal 1 (coordinator / master):

    ```bash
    scripts/run.sh run_master --log_num_workers $log_num_workers --log_num_vars $log_num_vars
    ```

	- When you see Master listening on ..., Terminal 2 (workers on same host):

    ```bash
    scripts/run.sh run_multi_worker --log_num_workers $log_num_workers --log_num_vars $log_num_vars
    ```

5.	Analyze

    ```bash
    scripts/run.sh analyze --log_num_workers $log_num_workers --log_num_vars $log_num_vars --analyze_target master
    ```

### B. Distributed on AWS (paper‑scale)

⚠️ Security note: for AE convenience we describe an “open” SG (All traffic / Anywhere). In production, restrict ports and source CIDRs. Ensure all hosts are in the same region/VPC and “Auto‑assign public IP” is enabled.

1.	Create an EC2 launch template

    - AMI: Ubuntu Server 24.04 LTS (HVM), SSD Volume Type
	- Instance type: t3.2xlarge
	- Storage: 50 GiB
	- Key pair: create an ed25519 key; save the private key locally as key/cirrus.pem
	- Security group: allow All traffic from Anywhere (AE only)
	- Advanced networking: enable Auto‑assign public IP

2.	Launch 33 instances from the template; record public IPs as:

    IP_0, IP_1, ..., IP_32

    IP_0 will be the coordinator (“master”).

3.	Prepare instances (from your local machine)

	- Create and activate a small Python env:

    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    source ~/miniconda3/bin/activate
    conda init --all

    conda create -n cirrus-ae python=3.12 -y
    conda activate cirrus-ae
    pip install paramiko
    ```

	- Place your private key at `key/cirrus.pem`.
	
    - Edit ec2/setup.py if needed (e.g., username, ports), then run:

    ```bash
    python ec2/setup.py
    ```

    This installs Rust and Python prerequisites on every instance.

4.	Upload setup (SRS) to instances

    If you downloaded or generated setup locally, configure and run:

    ```python
    # Edit ec2/transfer_setup_param.py:
    #   - set lists: log_num_workers = [ ... ], log_num_vars = [ ... ]
    #   - set ec2_public_ips = ["IP_0", "IP_1", ..., "IP_32"]
    ```

    ```bash
    python ec2/transfer_setup_param.py
    ```


## Setup parameters (universal SRS)

Cirrus reuses HyperPlonk’s universal trusted setup. You have three options:

1. **Authors prepare on servers (AE convenience)**: skip local generation and large transfers.
2. **Download (~450 GB)** and place under `out/`  
   Link: <https://drive.google.com/drive/folders/1ikHDad399ASC9f44J-oYQkvZHCRh4ual?usp=sharing>
3. **Generate locally** (slow; not part of the paper’s efficiency claims):
    ```bash
    scripts/run.sh setup --log_num_workers $log_num_workers --log_num_vars $log_num_vars
    # output → out/vanilla-$log_num_vars-$log_num_workers/
    ```

## Running experiments

### E1 — End-to-end proving time

Goal: reproduce the scalability figure; for claim (C1) set log_num_workers=8 and log_num_vars=25 (i.e., 256 workers, ~33 M gates).

Distributed run (from local machine)
1.	Edit ec2/run_exp_single_thread.py:
    ```python
    log_num_workers = [8]          # or a list, e.g., [5, 6, 7, 8]
    log_num_vars    = [25]         # or a list, e.g., [22, 23, 24, 25]

    master_ip = "IP_0"             # public IP of coordinator
    master_private_addr = "COORD_IP"  # coordinator's private IPv4 address
    master_listen_port = 7034

    worker_ips = [
    "IP_1", "IP_2", ..., "IP_32"
    ]
    ```


2.	Run:

    ```bash
    python ec2/run_exp_single_thread.py
    ```

### E2 — Accountability protocol runtime

Goal: show the coordinator can identify malicious workers quickly.

Run on the coordinator (IP_0):

```bash
./scripts/run.sh accountability \
  --log_num_vars $log_num_vars \
  --num_threads 8 \
  --log_num_workers $log_num_workers
```

The script prints:

```bash
[INFO] total time: {t} s
```

which is the accountability runtime for the chosen (log_num_workers, log_num_vars).

### E3 — Coordinator compute time (with/without HA)

Goal: demonstrate lightweight coordinator work and independence from sub-circuit size under HA.

Run on the coordinator (IP_0):

```bash
bash ./bench_master_ha.sh      # with hierarchical aggregation
bash ./bench_master_no_ha.sh   # without hierarchical aggregation
```

Each command prints lines like:

```bash
Time elapsed for --log-num-vars=$log_num_vars and --log-num-workers=$log_num_workers: {t}ms
```

## Acknowledgement

This project is supported in part by the Ethereum Foundation. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of these institutes.

## Disclaimer

This software is provided "as is." It has not undergone a third-party security audit.
Opening “All traffic / Anywhere” security groups is only for the convenience of AE and not recommended outside controlled environments. Use at your own risk.

