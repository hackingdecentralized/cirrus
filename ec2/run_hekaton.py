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
master_ip = "54.166.156.21"
master_private_addr = "172.31.119.134"
master_listen_port = 7034
worker_ips = [
    "3.84.242.172",
    "18.208.139.209",
    '34.204.18.137',
    '34.230.90.177',
    '107.21.71.201',
    '34.229.174.213',
    '18.212.85.139',
    '3.88.29.42',
    '3.88.176.34',
    '3.87.144.97',
    '34.230.76.160',
    '54.226.157.229',
    '54.226.180.80',
    '3.80.90.117',
    '3.93.173.183',
    '54.144.51.242',
    '3.90.183.81',
    '3.84.176.156',
    '184.73.132.182',
    '34.229.179.9',
    '54.196.137.12',
    '98.81.77.93',
    '107.21.180.96',
    '54.210.74.1',
    '98.84.116.117',
    '54.86.122.167',
    '54.91.62.79',
    '3.80.73.157',
    '18.212.75.158',
    '54.145.25.223',
    '54.221.170.67',
    '52.90.223.230'
]

# Define the configurations for log_num_workers and log_num_vars
num_subcircuits_values = [1024]
num_workers = [4]

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

# Function to create and upload the hosts file to the master node
def create_and_upload_hosts_file(master_ip, num_worker, worker_ips):
    hostfile_content = ["127.0.0.1 slots=1"]  # Add master node entry
    for i in range(min(num_worker, len(worker_ips))):
        hostfile_content.append(f"{worker_ips[i]} slots=1")
    
    # Write the hostfile locally
    hostfile_path = os.path.join(script_dir, "hosts")
    with open(hostfile_path, "w") as file:
        file.write("\n".join(hostfile_content))
    
    print(f"Hosts file created:\n{hostfile_content}")

    # Transfer the hostfile to the master node
    ssh_client = create_ssh_client(master_ip)
    if ssh_client:
        sftp = ssh_client.open_sftp()
        remote_hostfile_path = "/home/ubuntu/projects/hekaton-fork/hosts"
        sftp.put(hostfile_path, remote_hostfile_path)
        sftp.close()
        ssh_client.close()
        print(f"Hosts file transferred to {master_ip}:{remote_hostfile_path}")


# Function to run a command on a given IP in a separate thread
def run_command_in_thread(ip, command):
    ssh_client = create_ssh_client(ip)
    if ssh_client:
        execute_remote_command(ssh_client, command)
        ssh_client.close()

# Main function to run the test script
def run_test():
    for num_subcircuits in num_subcircuits_values:
        for num_worker in num_workers:
            if num_worker > num_subcircuits:
                continue
            
            create_and_upload_hosts_file(master_ip, num_worker, worker_ips)
            
            # Start mpi from master
            run_cmd = [
                f"source ~/.cargo/env && cd /home/ubuntu/projects/hekaton-fork && ./scripts/run.sh run --num_subcircuits {num_subcircuits} --num_workers {num_worker} --num_sha2_iters {1} --num_portals 7 --num_threads 8"
            ]
            
            thread = threading.Thread(target=run_command_in_thread, args=(master_ip, " && ".join(run_cmd)))
            thread.start()
            thread.join()
            
            print(f"Test for num_subcircuits={num_subcircuits}, num_workers={num_worker} complete.")

if __name__ == "__main__":
    run_test()