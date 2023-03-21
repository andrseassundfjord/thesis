from cmath import inf
import math
import numpy as np
import pandas as pd

class NN:

    def __init__(self, x, y, feature_name, feature_idx):
        self.x = x
        self.y = y
        self.feature_name = feature_name
        self.feature_idx = feature_idx

    def predict(self, x_index):
        """
        x_index: int, index of data you want to predict
        """
        nn = (0, inf)
        if self.feature_name not in self.x[x_index].columns:
            return None
        this_x = self.x[x_index][self.feature_name]
        for idx in range(len(self.x)):
            if self.feature_name not in self.x[idx].columns:
                continue
            diff = 0
            feature = self.x[idx][self.feature_name]
            for i, _ in enumerate(feature):
                val = 0
                if not math.isnan(feature[i]):
                    val = feature[i]
                diff += (this_x[i] - val)**2
            if diff < nn[1] and idx != x_index:
                nn = (idx, diff)
        return nn

    def get_error_rate(self):
        n_wrong_arr = np.zeros(16)
        accuracy_arr = np.zeros(16)
        fp_arr = np.zeros(16)
        for idx in range(len(self.x)):
            predicted = self.predict(idx)
            if predicted == None:
                continue
            predicted_idx = predicted[0]
            if self.y[idx] != self.y[predicted_idx] or predicted[1] == inf:
                class_idx = int(self.y[idx]) - 1
                predicted_class_idx = int(self.y[predicted_idx]) - 1
                n_wrong_arr[class_idx] += 1
                n_wrong_arr[14] += 1
                fp_arr[predicted_class_idx] += 1
                fp_arr[14] += 1
            else:
                class_idx = int(self.y[idx]) - 1
                accuracy_arr[class_idx] += 1
                accuracy_arr[14] += 1
        n_samples = pd.read_csv("results/class_statistics.csv")
        for idx, val in enumerate(n_wrong_arr):
            n = n_samples.sum()["class_counter"]
            if idx < 14:
                n = n_samples.iloc[idx,1]
            if idx == 15:
                n_wrong_arr[idx] = n_wrong_arr[idx-1] - round(n/14, 4)
            else:
                n_wrong_arr[idx] = round(val/n, 4)
        for idx, val in enumerate(accuracy_arr):
            n = n_samples.sum()["class_counter"]
            if idx < 14:
                n = n_samples.iloc[idx,1]
            if idx == 15:
                accuracy_arr[idx] = accuracy_arr[idx-1] - round(n/14, 4)
            else:
                accuracy_arr[idx] = round(val/n, 4)
        for idx, val in enumerate(fp_arr):
            n = n_samples.sum()["class_counter"]
            if idx < 14:
                n = n_samples.iloc[idx,1]
            if idx == 15:
                fp_arr[idx] = fp_arr[idx-1] - round(n/14, 4)
            else:
                fp_arr[idx] = round(val/n, 4)
        return n_wrong_arr, accuracy_arr, fp_arr