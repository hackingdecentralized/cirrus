mkdir -p ~/projects
rm -rf ~/projects/hekaton-fork
mv ~/hekaton.zip ~/projects/
cd ~/projects
unzip ~/projects/hekaton.zip -d ~/projects && rm ~/projects/hekaton.zip

sudo apt update
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
sudo apt install -y clang