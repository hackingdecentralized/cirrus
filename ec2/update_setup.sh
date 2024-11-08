# setup_commands.sh
rm -rf ~/projects
mkdir -p ~/projects
mv ~/cirrus.zip ~/projects/
cd ~/projects
sudo apt install unzip -y
unzip ~/projects/cirrus.zip -d ~/projects && rm ~/projects/cirrus.zip