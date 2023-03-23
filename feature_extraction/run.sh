#!/bin/bash

# Exit on failure
set -e

# activate virtual environment
source ../env/bin/activate

#python test.py
python run_MVAE.py
#python save_video.py