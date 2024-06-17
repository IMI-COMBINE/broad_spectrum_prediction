#!/usr/bin/env bash

# This script builds and installs TMAP (https://github.com/reymond-group/tmap). 
# Particully, this aims at ARM macOS, for which no official binary is provided.  

# Requires:
# Conda is needed to obtain the prerequisites: cmake and openmp. 

# To install:
# Place this script in a directory where you want to store the TMAP repo, then
# $ bash install_tmap.sh
 
# To use:
# This script installs tmap in a conda environment named `tmap-env`. 
# After installing, activate this conda environment to use it. 


# Achieved via the below steps:
# 0. clone the repo
# 1. creating an conda env named `tmap-env`
# 2. get `cmake` and `libomp` using conda
# 3. compile OGDF, which is shipped with tmap
# 4. export LIBOGDF_INSTALL_PATH
# 5. install tmap


# 0
GIT_REPO=https://github.com/reymond-group/tmap.git
git clone ${GIT_REPO}

# 1 and 2
source ${CONDA_PREFIX}/etc/profile.d/conda.sh # make sure conda activate works in a bash script
conda env remove --name tmap-env # remove env if it exists
conda create --name tmap-env -y
conda activate tmap-env
conda install -c conda-forge pip cmake llvm-openmp -y

# 3
cd tmap
pushd ogdf-conda/src/
mkdir build
mkdir installed
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=../installed
make -j 10
make install
popd

# 4
export LIBOGDF_INSTALL_PATH=$(pwd)/ogdf-conda/src/installed

# 5
# -I${CONDA_PREFIX}/include to use omp.h
CXXFLAGS=-I${CONDA_PREFIX}/include  pip install -e .