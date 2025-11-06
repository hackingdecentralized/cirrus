# setup_commands.sh
mkdir -p ~/projects
mv ~/cirrus.zip ~/projects/
cd ~/projects
sudo apt install unzip -y
unzip ~/projects/cirrus.zip -d ~/projects && rm ~/projects/cirrus.zip
chmod +x ~/projects/cirrus/scripts/run.sh

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
sudo apt update -y
sudo apt install -y build-essential

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all