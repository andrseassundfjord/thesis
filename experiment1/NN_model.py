from cmath import inf
import numpy as np

class NN:

    def __init__(self, x, y, dataset):
        self.x = x
        self.y = y
        self.dataset = dataset

    def predict(self, x_index):
        """
        x_index: int, index of data you want to predict
        features: list of ints, indices of features to include in prediction
        """
        nn = (0, inf)
        for idx in range(self.x.shape[0]):
            diff = 0
            for i in self.x[idx]:
                val = 0
                if not math.isnan(self.x[idx, i]):
                    val = self.x[idx, i]
                diff += (self.x[x_index, i] - val)**2
            if diff < nn[1] and idx != x_index:
                nn = (idx, diff)
        return nn

    def get_error_rate(self):
        n_wrong = 0
        for idx in range(self.x.shape[0]):
            predicted_idx = self.predict(idx, features)[0]
            if self.y[idx] != self.y[predicted_idx]:
                n_wrong += 1
        
        return round(n_wrong/self.x.shape[0], 4)