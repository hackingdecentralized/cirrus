import paramiko
from paramiko.ssh_exception import NoValidConnectionsError, SSHException
import threading
import time
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
master_dir = os.path.join(script_dir, "../")
master_dir = os.path.abspath(master_dir)

# Define the relative path to the key and resolve it to an absolute path
key_path = os.path.join(master_dir, "./key/cirrus.pem")
key_path = os.path.abspath(key_path)  # Resolve to an absolute path

# Define the public IP addresses of your EC2 instances
ec2_public_ips = [
    "3.133.95.167",
    "18.217.134.40",
    "18.216.185.101"
]

# Path to the bash file with commands
setup_cmd = os.path.join(script_dir, "update_setup.sh")

# List of files to be uploaded to specific instances
circuit_files = ["circuit.plonk", "master.pk", "verify.key"]
worker_files_pattern = "worker_{worker_id}.pk"

def transfer_circuit_files_single_thread(ip, main_key_path, log_num_vars, log_num_workers, instance_type, worker_id_start=0, num_threads=8):
    """Transfers circuit files to the master or worker instances."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username="ubuntu", key_filename=main_key_path)
        sftp = ssh.open_sftp()
        
        folder_path = f"out/vanilla-bn254-{log_num_vars}-{log_num_workers}"
        remote_out_path = f"/home/ubuntu/projects/cirrus/{folder_path}"
        ssh.exec_command(f"mkdir -p {remote_out_path}")

        if instance_type == "master":
            for file in circuit_files:
                local_path = os.path.join(script_dir, "../", folder_path, file)
                remote_path = os.path.join(remote_out_path, file)
                print(f"Transferring {file} to {ip}:{remote_path}...")
                sftp.put(local_path, remote_path)
        elif instance_type == "worker":
            for j in range(num_threads):  # Iterate to transfer files for each thread
                worker_id = j + worker_id_start
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

# Transfer circuit files
log_num_workers = [1, 2, 3, 4]
log_num_vars = [16, 17, 18, 19, 20]

for log_num_worker in log_num_workers:
    for log_num_var in log_num_vars:
        num_total_instances = 1 + ((1 << log_num_worker) + 7) // 8  # Total number of instances including the master
        threads = []
        for i, ip in enumerate(ec2_public_ips[:num_total_instances]):
            if i == 0:
                # Master instance
                thread = threading.Thread(target=transfer_circuit_files_single_thread, args=(ip, key_path, log_num_var, log_num_worker, "master"))
                thread.start()
                threads.append(thread)
            else:
                # Worker instances
                thread = threading.Thread(target=transfer_circuit_files_single_thread, args=(ip, key_path, log_num_var, log_num_worker, "worker", (i - 1)*8, min(1 << log_num_worker, 8)))
                thread.start()
                threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

# Wait for all circuit file transfers to complete
print("Circuit file transfer complete for all instances.")

print("Setup process complete for all instances.")
