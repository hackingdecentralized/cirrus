import paramiko
from paramiko.ssh_exception import NoValidConnectionsError, SSHException
import threading
import time
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the key and resolve it to an absolute path
key_path = os.path.join(script_dir, "../key/cirrus.pem")
key_path = os.path.abspath(key_path)  # Resolve to an absolute path

# Define the public IP addresses of your EC2 instances
ec2_public_ips = [
    "23.22.219.232",
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

# Path to the bash file with commands
setup_cmd = os.path.join(script_dir, "update_setup.sh")

# Local path to the `cirrus.zip` file
local_zip_path = "../cirrus.zip"

# List of files to be uploaded to specific instances
circuit_files = ["circuit.plonk", "master.pk", "verify.key"]
worker_files_pattern = "worker_{worker_id}.pk"

def load_commands_from_file(file_path):
    """Load commands from a bash file."""
    with open(file_path, 'r') as file:
        # Read lines, stripping comments and empty lines
        commands = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    return commands

# Load commands from the bash file
commands = load_commands_from_file(setup_cmd)

def transfer_repo_code(ip, main_key_path, local_zip_path):
    """Transfers a zip file containing the repo code to an EC2 instance."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f"Connecting to {ip} to transfer repo code...")
        ssh.connect(ip, username="ubuntu", key_filename=main_key_path)

        sftp = ssh.open_sftp()
        remote_zip_path = "/home/ubuntu/cirrus.zip"
        print(f"Transferring {local_zip_path} to {ip}:{remote_zip_path}...")
        sftp.put(local_zip_path, remote_zip_path)
        sftp.close()
        # Execute setup commands
        for command in commands:
            print(f"Executing: {command}")
            stdin, stdout, stderr = ssh.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()  # Wait for command to complete
            if exit_status == 0:
                print(f"Success: {command}")
            else:
                print(f"Error: {command}\n{stderr.read().decode()}")
            time.sleep(1)
    
        ssh.close()
        print(f"Repo code transfer complete on {ip}")
    except (SSHException, IOError) as e:
        print(f"Failed to transfer repo code on {ip}: {e}")

def transfer_circuit_files(ip, main_key_path, log_num_vars, log_num_workers, instance_type, worker_id=None):
    """Transfers circuit files to the master or worker instances."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username="ubuntu", key_filename=main_key_path)
        sftp = ssh.open_sftp()
        
        folder_path = f"out/vanilla-{log_num_vars}-{log_num_workers}"
        remote_out_path = f"/home/ubuntu/projects/cirrus/{folder_path}"
        ssh.exec_command(f"mkdir -p {remote_out_path}")

        if instance_type == "master":
            for file in circuit_files:
                local_path = os.path.join(script_dir, "../", folder_path, file)
                remote_path = os.path.join(remote_out_path, file)
                print(f"Transferring {file} to {ip}:{remote_path}...")
                sftp.put(local_path, remote_path)
        elif instance_type == "worker" and worker_id is not None:
            worker_file = worker_files_pattern.format(worker_id=worker_id)
            local_path = os.path.join(script_dir, "../", folder_path, worker_file)
            remote_path = os.path.join(remote_out_path, worker_file)
            print(f"Transferring {worker_file} to {ip}:{remote_path}...")
            sftp.put(local_path, remote_path)
        
        sftp.close()
        ssh.close()
        print(f"Circuit files transfer complete on {ip}")
    except (SSHException, IOError) as e:
        print(f"Failed to transfer circuit files on {ip}: {e}")

# Run the setup process on each EC2 instance in parallel
# threads = []
# master_ip = ec2_public_ips[0]  # First IP is the master

# # Transfer the repo code to all instances
# for ip in ec2_public_ips:
#     thread = threading.Thread(target=transfer_repo_code, args=(ip, key_path, local_zip_path))
#     thread.start()
#     threads.append(thread)

# # Wait for all repo code transfers to complete
# for thread in threads:
#     thread.join()

# print("Repo code transfer complete for all instances.")

# Transfer circuit files
num_workers = [1, 2, 3, 4, 5]
log_num_vars = [16, 17, 18, 19, 20, 21, 22]

for log_num_worker in num_workers:
    for log_num_var in log_num_vars:
        threads = []
        for i, ip in enumerate(ec2_public_ips[:1 + (1 << log_num_worker)]):
            if i == 0:
                # Master instance
                thread = threading.Thread(target=transfer_circuit_files, args=(ip, key_path, log_num_var, log_num_worker, "master"))
            else:
                # Worker instances
                worker_id = i - 1
                thread = threading.Thread(target=transfer_circuit_files, args=(ip, key_path, log_num_var, log_num_worker, "worker", worker_id))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

# Wait for all circuit file transfers to complete


print("Circuit file transfer complete for all instances.")

print("Setup process complete for all instances.")
