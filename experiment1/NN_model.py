from cmath import inf
import numpy as np

class NN:

    def __init__(self, x, y, dataset):
        self.x = x
        self.y = y
        self.dataset = dataset

    def predict(self, x_index, features):
        """
        x_index: int, index of data you want to predict
        features: list of ints, indices of features to include in prediction
        """
        nn = (0, inf)
        for idx in range(self.x.shape[0]):
            diff = 0
            for feature_idx in features:
                diff += (self.x[x_index, feature_idx] - self.x[idx, feature_idx])**2
            if diff < nn[1] and idx != x_index:
                nn = (idx, diff)
        return nn

    def get_error_rate(self, features):
        n_wrong = 0
        for idx in range(self.x.shape[0]):
            predicted_idx = self.predict(idx, features)[0]
            if self.y[idx] != self.y[predicted_idx]:
                n_wrong += 1
        
        return round(n_wrong/self.x.shape[0], 4)
    
    def save_results(self):
        num_features = self.x.shape[1]
        file = open("results/NN_{}.txt".format(self.dataset), "w")
        for i in range(num_features):
            # Test only one feature
            er = self.get_error_rate([i])
            file.write("Feature {} error rate: {} \n".format(i+1, er))
            for j in range(num_features):
                # Test two features
                if i < j:
                    er = self.get_error_rate([i, j])
                    file.write("Feature {} and {} error rate: {} \n".format(i+1, j+1, er))  
                    if num_features == 4:
                        for k in range(num_features):
                            # Test three features
                            if j < k:
                                er = self.get_error_rate([i, j, k])
                                file.write("Feature {}, {} and {} error rate: {} \n".format(i+1, j+1, k+1, er))
        # Test all features
        er = self.get_error_rate(range(num_features))
        file.write("Features {}-{} error rate: {}".format(1, num_features, er))
        file.close()