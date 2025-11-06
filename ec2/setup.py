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
        # "3.133.95.167",
    "18.217.134.40",
    "18.216.185.101"
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
