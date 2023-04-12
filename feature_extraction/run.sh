#!/bin/bash

# Exit on failure
set -e
export CUDA_LAUNCH_BLOCKING=1
set CUDA_LAUNCH_BLOCKING=1

# activate virtual environment
source ../env/bin/activate

#python test.py
python run_MVAE.py
#python save_video.py
#python load_dataset.py

unset CUDA_LAUNCH_BLOCKING
