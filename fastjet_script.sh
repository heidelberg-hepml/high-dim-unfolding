#!/bin/bash
cd /path/to/your/project
source .venv/bin/activate

mkdir fastjet
cd fastjet
mkdir fastjet-install

FASTJET_PATH=$PWD/fastjet-install
FASTJET_VERSION=3.4.3
FJCONTRIB_VERSION=1.055

wget https://fastjet.fr/repo/fastjet-$FASTJET_VERSION.tar.gz
tar zxvf fastjet-$FASTJET_VERSION.tar.gz
cd fastjet-$FASTJET_VERSION/
./configure --prefix=$FASTJET_PATH --enable-allcxxplugins --disable-auto-ptr CXXFLAGS=-fPIC
make CXXFLAGS="-fPIC"
make install CXXFLAGS="-fPIC"

cd ..

wget https://fastjet.fr/contrib/downloads/fjcontrib-$FJCONTRIB_VERSION.tar.gz
tar zxvf fjcontrib-$FJCONTRIB_VERSION.tar.gz
cd fjcontrib-$FJCONTRIB_VERSION
./configure --fastjet-config=$FASTJET_PATH/bin/fastjet-config CXXFLAGS=-fPIC
make
make install
make fragile-shared
make fragile-shared-install

cd ..
export LD_LIBRARY_PATH=$FASTJET_PATH/lib:$LD_LIBRARY_PATH

git clone https://github.com/AntoinePTJ/pybind_fastjet_contribs.git
cd pybind_fastjet_contribs
sed -i "7s|.*|fastjet_dir = '$FASTJET_PATH'|" setup.py
uv pip install .