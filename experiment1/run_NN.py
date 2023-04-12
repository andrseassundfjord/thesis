import os
import pandas as pd
import numpy as np
from NN_model import NN
import seaborn as sns

def get_dataset(path, files):
       all_x = []
       y = []
       for filename in files:
              df = pd.read_pickle(path + "/" + filename)
              all_x.append(df) 
              label = filename.split(".")[0].split("_")[-1]
              y.append(label)
       return all_x, np.array(y)

def run(test = True):
       path_to_data = "resampled_pickles"
       files = os.listdir(path_to_data)
       feature_stats = pd.read_csv("results/feature_statistics.csv")
       features = feature_stats.columns[1:]
       nn_results_er = pd.DataFrame(index = features)
       nn_results_acc = pd.DataFrame(index = features)
       nn_results_fp = pd.DataFrame(index = features)
       x, y = get_dataset(path_to_data, files)
       for i in range(1, 17):
              nn_results_er[str(i)] = np.zeros(len(features))
              nn_results_acc[str(i)] = np.zeros(len(features))
              nn_results_fp[str(i)] = np.zeros(len(features))
       print("Started")
       for idx, feature in enumerate(features):
              if feature == "Unnamed: 0" or "mobileye" in feature:
                     nn_results_er.drop(feature)
                     nn_results_acc.drop(feature)
                     nn_results_fp.drop(feature)
                     continue
              print("Feature: ", feature, flush = True)
              nn = NN(x, y, feature, idx)
              error_rate_arr, accuracy_arr, fp_arr = nn.get_error_rate()
              print("error rate: ", error_rate_arr, flush=True)
              print("accuracy: ", accuracy_arr, flush=True)
              print("false positive: ", fp_arr, flush=True)
              for i in range(1, 17):
                     nn_results_er[str(i)][feature] = error_rate_arr[i-1]
                     nn_results_acc[str(i)][feature] = accuracy_arr[i-1]
                     nn_results_fp[str(i)][feature] = fp_arr[i-1]
              if test:
                     nn_results_er.to_csv("results/nn_test_er.csv")
                     nn_results_acc.to_csv("results/nn_test_acc.csv")
                     nn_results_fp.to_csv("results/nn_test_fp.csv")
                     return
       nn_results_er.to_csv("results/nn_result_er.csv")
       nn_results_acc.to_csv("results/nn_result_acc.csv")
       nn_results_fp.to_csv("results/nn_result_fp.csv")
       print("Saved files")
       plot_results()
       print("Plotted results")

def plot_results():
       # Plot error rate
       er = pd.read_csv("results/nn_result_er.csv")
       er_fig = sns.heatmap(er).figure
       er_fig.savefig("results/error_rate_nn")
       # Plot accuracy
       acc = pd.read_csv("results/nn_result_acc.csv")
       acc_fig = sns.heatmap(er).figure
       acc_fig.savefig("results/accuracy_nn")
       # Plot false positives
       fp = pd.read_csv("results/nn_result_fp.csv")
       acc_fig = sns.heatmap(er).figure
       acc_fig.savefig("results/false_positives_nn")
    
if __name__ == "__main__":
       run(test = False)