sudo apt-get install bison
bash < <(curl -s -S -L https://raw.githubusercontent.com/moovweb/gvm/master/binscripts/gvm-installer)
gvm install --prefer-binary go1.20.5
gvm use go1.20.5 --default

cd ~/projects && git clone https://github.com/dreamATD/pianist-gnark.git
