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
    "3.94.193.121",
    "54.198.178.147",
    "52.54.158.140",
    "54.88.122.210",
    "3.88.185.136",
    "98.81.154.255",
    "54.147.162.36",
    "98.81.211.140",
    "54.242.169.47",
    "107.22.150.115",
    "34.207.182.35",
    "54.157.250.164",
    "34.201.120.122",
    "54.205.79.20",
    "54.234.158.153",
    "54.227.100.237",
    "54.172.50.9",
    "54.242.82.162",
    "54.197.4.244",
    "3.95.238.147",
    "52.204.108.92",
    "3.90.18.206",
    "34.202.230.52",
    "52.90.144.96",
    "54.221.158.147",
    "54.163.196.198",
    "54.227.21.214",
    "52.201.214.107",
    "174.129.189.164",
    "54.82.21.167",
    "52.91.8.13",
    "52.91.51.190"
]

# Define the configurations for log_num_workers and log_num_vars
num_subcircuits_values = [4, 8, 16, 32, 64, 128, 256, 512]
num_workers = [2, 4, 8, 16, 32]

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