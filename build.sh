#!/bin/bash
#
# Build a virtual environment to get start 

# create virtual environment
rm -rf .venv
rm -rf __pycache__
python3 -m venv .venv

# active virtual environment
source .venv/bin/activate

# install requirements
python3 -m pip install -r requirements.txt

# exit virtual environment
# deactivate