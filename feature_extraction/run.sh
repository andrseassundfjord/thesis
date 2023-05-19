#!/bin/bash

# Exit on failure
set -e
export CUDA_LAUNCH_BLOCKING=1
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# activate virtual environment
source ../env/bin/activate

#python test.py
#python run_MVAE.py
#python run_MVAE_augmented.py
#python save_video.py
python plot_timeseries.py
#python classification.py
#python clustering.py
#python gmm.py
#python risk_prediction.py
#python load_dataset.py

unset CUDA_LAUNCH_BLOCKING
