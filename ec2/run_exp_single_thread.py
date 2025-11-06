import paramiko
from paramiko.ssh_exception import NoValidConnectionsError, SSHException
import threading
import time
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the key and resolve it to an absolute path
key_path = os.path.join(script_dir, "../key/cirrus.pem")
key_path = os.path.abspath(key_path)  # Resolve to an absolute path

# Define the public IP addresses of master and worker EC2 instances
master_ip = "3.133.95.167"
# The private IPV4 address
master_private_addr = "172.31.19.161"
master_listen_port = 7034
worker_ips = [
    "18.217.134.40",
    "18.216.185.101"
]

# Define the configurations for log_num_workers and log_num_vars
log_num_workers_configs = [4]  # Example configurations for log_num_workers
log_num_vars_configs = [16]  # Example configurations for log_num_vars
curve = "bls12_381"

# Function to create SSH client
def create_ssh_client(ip):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(ip, username='ubuntu', key_filename=key_path)
        return client
    except (NoValidConnectionsError, SSHException) as e:
        print(f"Failed to connect to {ip}: {e}")
        return None

# Function to execute commands on a remote host
def execute_remote_command(ssh_client, command):
    try:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        print(f"Output for command '{command}':\n{stdout.read().decode()}")
        print(f"Error for command '{command}':\n{stderr.read().decode()}")
        sys.stdout.flush()
    except SSHException as e:
        print(f"Failed to execute command '{command}': {e}")
        sys.stdout.flush()

# Function to run a command on a given IP in a separate thread
def run_command_in_thread(ip, command):
    ssh_client = create_ssh_client(ip)
    if ssh_client:
        execute_remote_command(ssh_client, command)
        ssh_client.close()

# Main function to run the test script
def run_test():
    for log_num_workers in log_num_workers_configs:
        for log_num_vars in log_num_vars_configs:
            print(f"Running test with log_num_workers={log_num_workers}, log_num_vars={log_num_vars}")

            # Calculate the number of workers needed based on log_num_workers
            num_workers = 1 << log_num_workers  # 2^log_num_workers
            selected_worker_ips = worker_ips[:(num_workers + 7) // 8]  # Select the required number of worker IPs, rounding up to allocate one machine per 8 workers

            # Check if we have enough worker IPs
            if len(selected_worker_ips) < (num_workers + 7) // 8:
                print(f"Not enough worker IPs. Required: {(num_workers + 7) // 8}, Available: {len(selected_worker_ips)}")
                continue

            # Start master node in a separate thread
            run_master_cmd = [
                f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh run_master --log_num_workers {log_num_workers} --log_num_vars {log_num_vars} --master_addr {master_private_addr}:{master_listen_port} --curve {curve}"
            ]
            master_thread = threading.Thread(target=run_command_in_thread, args=(master_ip, " && ".join(run_master_cmd)))
            master_thread.start()

            # Wait briefly before setting up and starting workers in parallel
            time.sleep(1 << (max(1, log_num_vars - 14)))
            
            worker_threads = []
            for j, worker_ip in enumerate(selected_worker_ips):
                commands = []
                for i in range(min(8, 1 << log_num_workers)):  # Each worker machine runs up to 8 worker programs
                    worker_id = j * 8 + i
                    run_worker_cmd = (
                        f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh run_single_worker "
                        f"--num_threads 1 --log_num_workers {log_num_workers} --log_num_vars {log_num_vars} "
                        f"--master_addr {master_private_addr}:{master_listen_port} --worker_id {worker_id} --curve {curve}"
                    )
                    commands.append(run_worker_cmd)
            
                # Join commands to run concurrently in a single SSH connection
                combined_command = " & ".join(commands)
                worker_thread = threading.Thread(target=run_command_in_thread, args=(worker_ip, combined_command))
                worker_thread.start()
                worker_threads.append(worker_thread)
                time.sleep(1)

            # Wait for all threads (master and workers) to complete
            master_thread.join()
            for worker_thread in worker_threads:
                worker_thread.join()

            # Start analyze command in parallel
            analyze_master_cmd = [
                f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh analyze --log_num_workers {log_num_workers} --log_num_vars {log_num_vars} --analyze_target master"
            ]
            master_analyze_thread = threading.Thread(target=run_command_in_thread, args=(master_ip, " && ".join(analyze_master_cmd)))
            master_analyze_thread.start()

            worker_analyze_threads = []
            for worker_id, worker_ip in enumerate(selected_worker_ips):
                analyze_commands = []
                for i in range(min(8, 1 << log_num_workers)):  # Analyze each worker program
                    analyze_worker_cmd = (
                        f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh analyze "
                        f"--log_num_workers {log_num_workers} --log_num_vars {log_num_vars} "
                        f"--analyze_target worker_{worker_id * 8 + i}"
                    )
                    analyze_commands.append(analyze_worker_cmd)

                # Join analyze commands to run concurrently in a single SSH connection
                combined_analyze_command = " & ".join(analyze_commands) + " & wait"
                worker_analyze_thread = threading.Thread(target=run_command_in_thread, args=(worker_ip, combined_analyze_command))
                worker_analyze_thread.start()
                worker_analyze_threads.append(worker_analyze_thread)

            master_analyze_thread.join()
            for worker_analyze_thread in worker_analyze_threads:
                worker_analyze_thread.join()

            print(f"Completed test with log_num_workers={log_num_workers}, log_num_vars={log_num_vars}")

if __name__ == "__main__":
    run_test()
