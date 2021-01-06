#!/bin/bash
#
# Build a virtual environment to get start 

# create virtual environment
rm -rf .venv
rm -rf __pycache__
python3 -m venv .venv

# active virtual environment
source .venv/bin/activate

#
# Workaround to the [#4919](https://github.com/scikit-image/scikit-image/issues/4919)
# ModuleNotFoundError: No module named 'numpy'
# when install scikit-image
#
# 06/01/2021
python3 -m pip install numpy==1.19.4

# install requirements
python3 -m pip install -r requirements.txt

# exit virtual environment
deactivate