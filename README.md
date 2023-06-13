# Master's thesis by Andreas Sundfjord

The script for training the models is in run_MVAE_augmented.py. The argument for which model to train is set
in the script, together with arguments for hyperparameters and whether to load the dataset.
The script also runs all evaluation tasks, which are in classification.py, gmm.py, clustering.py, 
and risk_prediction.py. 

Each model is implemented in the feature_extraction folder.