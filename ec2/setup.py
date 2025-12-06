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

# Path to the bash file with commands
setup_cmd = os.path.join(script_dir, "setup.sh")

# Local path to the `cirrus.zip` file
local_zip_path = os.path.join(master_dir, "cirrus.zip")
remote_zip_path = "/home/ubuntu/cirrus.zip"

# Define the public IP addresses of your EC2 instances
ec2_public_ips = [
    "3.150.127.61", # Coordinator, with private IP 172.31.30.68
    "18.221.126.43",
    "18.221.38.174",
    "3.22.95.243",
    "18.219.222.16",
    "18.117.196.192",
    "18.220.174.152",
    "3.15.39.183",
    "3.17.135.22",
    "18.222.137.234",
    "3.129.44.207",
    "3.21.240.30",
    "3.140.200.134",
    "18.218.220.85",
    "18.217.223.136",
    "3.145.211.4",
    "3.17.132.163",
    "3.145.216.195",
    "3.14.147.71",
    "18.188.67.217",
    "18.221.113.66",
    "18.118.194.17",
    "18.116.97.127",
    "52.15.217.53",
    "3.135.222.168",
    "13.58.204.23",
    "18.222.204.15",
    "3.145.113.6",
    "18.218.251.11",
    "3.128.192.215",
    "18.222.210.174",
    "3.128.180.64",
    "18.191.167.73"
]

def load_commands_from_file(file_path):
    """Load commands from a bash file."""
    with open(file_path, 'r') as file:
        # Read lines, stripping comments and empty lines
        commands = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    return commands

# Load commands from the bash file
commands = load_commands_from_file(setup_cmd)

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
threads = []
for ip in ec2_public_ips:
    thread = threading.Thread(target=transfer_and_setup_repo, args=(ip, key_path, local_zip_path, commands))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("Setup process complete for all instances.")
