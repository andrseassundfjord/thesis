#!/bin/bash

# Exit on failure
set -e

# activate virtual environment
source ../env/bin/activate

#python test.py
#python dataloader.py
python run_NN.py