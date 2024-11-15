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
    "54.166.156.21",
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
    '52.90.223.230',
]

ec2_public_ips_tmp = ["54.198.178.147",]

# Path to the bash file with commands
setup_cmd = os.path.join(script_dir, "setup_hekaton.sh")

# Local path to the `cirrus.zip` file
local_zip_path = "../hekaton.zip"
remote_zip_path = "/home/ubuntu/hekaton.zip"

# def load_commands_from_file(file_path):
#     """Load commands from a bash file."""
#     with open(file_path, 'r') as file:
#         # Read lines, stripping comments and empty lines
#         commands = [line.strip() for line in file if line.strip() and not line.startswith("#")]
#     return commands

# # Load commands from the bash file
# commands = load_commands_from_file(setup_cmd)
# commands = ["sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1"]

def transfer_and_setup_repo(ip, main_key_path, local_zip_path, commands):
    """Transfers a zip file and sets up the repository on an EC2 instance."""
    try:
        # Create an SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the EC2 instance
        print(f"Connecting to {ip}...")
        ssh.connect(ip, username="ubuntu", key_filename=main_key_path)

        # Open an SFTP session and transfer the zip file
        sftp = ssh.open_sftp()
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
        print(f"Repository setup complete on {ip}")
    
    except (SSHException, IOError) as e:
        print(f"Failed to setup repo on {ip}: {e}")

# Run the setup process on each EC2 instance in parallel
# threads = []
# for ip in ec2_public_ips_tmp:
#     thread = threading.Thread(target=transfer_and_setup_repo, args=(ip, key_path, local_zip_path, commands))
#     thread.start()
#     threads.append(thread)

# # Wait for all threads to complete
# for thread in threads:
#     thread.join()

print("Setup process complete for all instances.")

# Function to transfer circuit files
def transfer_circuit_files(ip, main_key_path):
    """Transfers circuit files based on num_subcircuits and num_total to the master or worker instances."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username="ubuntu", key_filename=main_key_path)
        sftp = ssh.open_sftp()
        
        num_subcircuits_values = [1024, 2048]
        # local_path =["/home/wenhaowang/projects/cirrus/cirrus/bin/hyperplonk.rs",
        #              "/home/wenhaowang/projects/cirrus/cirrus/bin/master.rs",
        #              "/home/wenhaowang/projects/cirrus/cirrus/bin/setup.rs",
        #              "/home/wenhaowang/projects/cirrus/cirrus/bin/worker.rs",
        #              "/home/wenhaowang/projects/cirrus/cirrus/Cargo.toml",
        #              "/home/wenhaowang/projects/cirrus/scripts/run.sh"]
        # remote_path = ["/home/ubuntu/projects/cirrus/cirrus/bin/hyperplonk.rs",
        #              "/home/ubuntu/projects/cirrus/cirrus/bin/master.rs",
        #              "/home/ubuntu/projects/cirrus/cirrus/bin/setup.rs",
        #              "/home/ubuntu/projects/cirrus/cirrus/bin/worker.rs",
        #              "/home/ubuntu/projects/cirrus/cirrus/Cargo.toml",
        #              "/home/ubuntu/projects/cirrus/scripts/run.sh"]
        # for path1, path2 in zip(local_path, remote_path):
        #     print(f"Transferring {path1} to {ip}:{path2}...")
        #     sftp.put(path1, path2)

        for num_subcircuits in num_subcircuits_values:
            num_sha2_iters = 1
            folder_path = f"/home/wenhaowang/projects/hekaton-fork/out/pks-big-merkle-{num_subcircuits}_{num_sha2_iters}_7"
            remote_out_path = f"/home/ubuntu/projects/hekaton-fork/out/pks-big-merkle-{num_subcircuits}_{num_sha2_iters}_7"
            ssh.exec_command(f"mkdir -p {remote_out_path}")

            for file in os.listdir(folder_path):
                if file.endswith("key.bin"):
                    local_path = os.path.join(folder_path, file)
                    remote_path = os.path.join(remote_out_path, file)
                    print(f"Transferring {local_path} to {ip}:{remote_path}...")
                    sftp.put(local_path, remote_path)
        
        sftp.close()
        ssh.close()
        print(f"Circuit files transfer complete on {ip}")
    except (SSHException, IOError) as e:
        print(f"Failed to transfer circuit files on {ip}: {e}")

# Run the transfer process on each EC2 instance in parallel
threads = []
for ip in ec2_public_ips:
    thread = threading.Thread(target=transfer_circuit_files, args=(ip, key_path))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("key file transfer complete for all instances.")
