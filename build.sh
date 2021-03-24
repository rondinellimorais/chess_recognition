#!/bin/bash
#
# Build a virtual environment to get start

#
# USAGE
#
# Esse script deve ser executado assim `source build.sh`, caso contrário
# não vamos conseguir ativar o ambiente conda.
#
# https://stackoverflow.com/questions/55507519/python-activate-conda-env-through-shell-script
#

# create virtual environment
rm -rf __pycache__
conda create -n chess_recognition python=3.9 pip --yes

# active virtual environment
conda activate chess_recognition

# install requirements
pip install -r requirements.txt