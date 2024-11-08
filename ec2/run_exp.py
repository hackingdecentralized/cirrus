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
master_ip = "23.22.219.232"
master_addr = "172.31.119.134:7021"
worker_ips = [
    "34.230.29.94",
    "34.228.79.107",
    '54.144.50.197',
    '23.22.232.175',
    '54.87.4.182',
    '98.80.8.31',
    '54.152.74.154',
    '54.164.223.244',
    '3.91.190.135',
    '54.87.27.71',
    '54.196.217.135',
    '54.167.4.15',
    '34.236.155.14',
    '34.227.226.47',
    '54.210.54.145',
    '34.224.166.48',
    '3.80.195.113',
    '54.160.211.46',
    '54.167.0.186',
    '54.196.4.150',
    '54.221.18.232',
    '107.20.128.104',
    '52.23.234.30',
    '3.89.211.109',
    '3.80.75.193',
    '3.91.16.200',
    '52.72.195.18',
    '54.226.5.208',
    '54.87.223.134',
    '54.242.56.55',
    '34.207.210.134',
    '54.235.232.106',
]

# Define the configurations for log_num_workers and log_num_vars
log_num_workers_configs = [4, 3, 2, 1]  # Example configurations for log_num_workers
log_num_vars_configs = [16, 17, 18, 19, 20, 21, 22]  # Example configurations for log_num_vars

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

            # Calculate the number of workers to use based on log_num_workers
            num_workers = 1 << log_num_workers  # 2^log_num_workers
            selected_worker_ips = worker_ips[:num_workers]  # Select the required number of worker IPs

            # Check if we have enough worker IPs
            if len(selected_worker_ips) < num_workers:
                print(f"Not enough worker IPs. Required: {num_workers}, Available: {len(selected_worker_ips)}")
                continue

            # Setup command for master and each worker with sourcing cargo environment
            setup_cmd = f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh setup --log_num_workers {log_num_workers} --log_num_vars {log_num_vars}"
            
            # # Start setup threads for both master and workers
            # setup_threads = []
            
            # # Master setup thread
            # master_thread = threading.Thread(target=run_command_in_thread, args=(master_ip, setup_cmd))
            # setup_threads.append(master_thread)
            # master_thread.start()
            
            # # Worker setup threads
            # for worker_ip in selected_worker_ips:
            #     worker_thread = threading.Thread(target=run_command_in_thread, args=(worker_ip, setup_cmd))
            #     setup_threads.append(worker_thread)
            #     worker_thread.start()

            # # Wait for all setup threads to complete
            # for thread in setup_threads:
            #     thread.join()

            # Start master node in a separate thread
            run_master_cmd = [
                f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh run_master --log_num_workers {log_num_workers} --log_num_vars {log_num_vars} --master_addr {master_addr}"
            ]
            master_thread = threading.Thread(target=run_command_in_thread, args=(master_ip, " && ".join(run_master_cmd)))
            master_thread.start()

            # Wait briefly before setting up and starting workers in parallel
            time.sleep(1 << (max(1, log_num_vars - 14)))
            
            # Start each worker node run command in parallel
            worker_threads = []
            for worker_id, worker_ip in enumerate(selected_worker_ips):
                # Run command for each worker
                run_worker_cmd = [
                    f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh run_single_worker --log_num_workers {log_num_workers} --log_num_vars {log_num_vars} --master_addr {master_addr} --worker_id {worker_id}"
                ]
                worker_thread = threading.Thread(target=run_command_in_thread, args=(worker_ip, " && ".join(run_worker_cmd)))
                worker_thread.start()
                worker_threads.append(worker_thread)
                time.sleep(1 << (max(1, log_num_vars - 19)))

            # Wait for all threads (master and workers) to complete
            master_thread.join()
            for worker_thread in worker_threads:
                worker_thread.join()
            
            # start analyze command in parallel
            analyze_master_cmd = [
                f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh analyze --log_num_workers {log_num_workers} --log_num_vars {log_num_vars} --master_addr {master_addr} --analyze_target master"
            ]
            master_analyze_thread = threading.Thread(target=run_command_in_thread, args=(master_ip, " && ".join(analyze_master_cmd)))
            master_analyze_thread.start()
            
            worker_analyze_threads = []
            for worker_id, worker_ip in enumerate(selected_worker_ips):
                analyze_worker_cmd = [
                    f"source ~/.cargo/env && cd /home/ubuntu/projects/cirrus && ./scripts/run.sh analyze --log_num_workers {log_num_workers} --log_num_vars {log_num_vars} --master_addr {master_addr} --analyze_target worker_{worker_id}"
                ]
                worker_analyze_thread = threading.Thread(target=run_command_in_thread, args=(worker_ip, " && ".join(analyze_worker_cmd)))
                worker_analyze_thread.start()
                worker_analyze_threads.append(worker_analyze_thread)
            
            master_analyze_thread.join()
            for worker_analyze_thread in worker_analyze_threads:
                worker_analyze_thread.join()

            print(f"Completed test with log_num_workers={log_num_workers}, log_num_vars={log_num_vars}")

if __name__ == "__main__":
    run_test()
