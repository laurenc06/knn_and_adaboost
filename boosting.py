#!/usr/bin/env python3
# Lauren Cho
# A2N1T01 

"""
CMPSC 165B - Machine Learning
Homework 3, Problem 2: Boosting Classifier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    # first, read the features file into an array
    X_df = pd.read_csv(X_path, skipinitialspace=True)
    X = X_df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    if y_path is None:
        return X

    y_df = pd.read_csv(y_path, skipinitialspace=True)  # header auto-detected
    y = y_df.iloc[:, 0].apply(pd.to_numeric, errors="raise").to_numpy(dtype=int)

    # searched this up but it basically lalows us to flatten it without copying memory
    y = y.ravel()

    return X, y


def preprocess_data(X_train, X_test):
    """Preprocess training and test data."""
    # standardize data so that they are all roughly the same size
    # 1) compute per-column mean on X_train
    mean = X_train.mean(axis=0)

    # 2) compute per-column std on X_train
    std = X_train.std(axis=0)

    # 3) avoid division by zero for std
    std[std == 0] = 1.0

    # 4) standardize X_train and X_test using those stats
    X_train_new = (X_train - mean)/std
    X_test_new = (X_test - mean)/std

    if np.isnan(X_train_new).any() or np.isnan(X_test_new).any():
        raise ValueError("NaNs found after preprocessing — data loading/conversion is wrong.")
  
    return X_train_new, X_test_new


class BoostingClassifier:
    """AdaBoost Classifier with weighted linear classifier as weak learner."""
    # initialize our stuff
    def __init__(self, T=10):
        self.T = T
        self.learners = []
        self.alphas = []

    # helper function for each of the learners
    def learner_help(self, X, y, sample_weights):
        pos_mask = (y == 1)
        neg_mask = (y == -1)

        X_pos = X[pos_mask]
        X_neg = X[neg_mask]

        w_pos = sample_weights[pos_mask]
        w_neg = sample_weights[neg_mask]

        num_features = X.shape[1]

        centroid_pos = np.zeros(num_features)
        centroid_neg = np.zeros(num_features)

        sum_w_pos = 0.0
        sum_w_neg = 0.0

        # positive class centroid
        for i in range(len(X_pos)):
            sum_w_pos += w_pos[i]
            for j in range(num_features):
                centroid_pos[j] += w_pos[i] * X_pos[i][j]

        for j in range(num_features):
            centroid_pos[j] /= sum_w_pos

        # negative class centroid
        for i in range(len(X_neg)):
            sum_w_neg += w_neg[i]
            for j in range(num_features):
                centroid_neg[j] += w_neg[i] * X_neg[i][j]

        for j in range(num_features):
            centroid_neg[j] /= sum_w_neg

        midpoint = (centroid_pos + centroid_neg)/2.0
        direction = centroid_pos - centroid_neg

        return {"centroid_pos": centroid_pos, "centroid_neg": centroid_neg, "midpoint": midpoint, "direction": direction, }


    def learner_predict(self, learner, X):
        scores = np.dot(X - learner["midpoint"], learner["direction"])
        return np.where(scores > 0, 1, -1)
    
    def train(self, X, y):
        """Fit the classifier to training data."""
        # Initialize : w_i = 1/ m for all i = 1,..,m (m = number of training points)
        m = X.shape[0] # this 
        w = np.ones(m) / m

        self.learners = []
        self.alphas = []

        for t in range(self.T):
            # Train weak learner h_t using weights w
            learner  = self.learner_help(X, y, w)
            pred = self.learner_predict(learner, X)

            # Compute weighted error
            incorrect = (pred != y)
            epsilon = np.sum(w[incorrect])

            # to ensure no division by zero
            epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)

            # Compute model weight
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)

            # Update sample weights :
            # - For misclassified points : w_i *= exp ( alpha_t )
            # - For correctly classified points : w_i *= exp ( - alpha_t )
            w[incorrect] *= np.exp(alpha)
            w[~incorrect] *= np.exp(-alpha)

            # Normalize weights so they sum to 1
            w /= np.sum(w)

            self.learners.append(learner)
            self.alphas.append(alpha)

    def predict(self, X):
        """Predict labels for input samples."""
        final_scores = np.zeros(X.shape[0])

        for alpha, learner in zip(self.alphas, self.learners):
            pred = self.learner_predict(learner, X)
            final_scores += alpha * pred

        return np.where(final_scores > 0, 1, -1)


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    # accuracy is fraction of predictions that equal the true label
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # np.mean treats true as 1 and false as 0
    return np.mean(y_true == y_pred)


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""
    # Load train
    X_train, y_train = load_data(Xtrain_file, Ytrain_file)

    # Load test
    X_test = load_data(test_data_file)

    # Preprocess
    X_train_p, X_test_p = preprocess_data(X_train, X_test)

    # temp prints for testing
    # print("X_train", X_train.shape)
    # print("y_train", y_train.shape)
    # print("X_test", X_test.shape)

    # Train
    model = BoostingClassifier(T=10)  # tune T
    model.train(X_train_p, y_train)

    # Predict
    y_pred = model.predict(X_test_p).astype(int)

    # Save one integer per line, no header according to assignment instructions
    np.savetxt(pred_file, y_pred, fmt="%d")

# This is a block of code for my testing and for generating my grpahs for the report
if __name__ == "__main__":
    X, y = load_data("wine_X.csv", "wine_y.csv")

    n = len(X)
    split_90 = int(0.9 * n)

    X_train_full = X[:split_90]
    y_train_full = y[:split_90]

    X_test = X[split_90:]
    y_test = y[split_90:]

    # Preprocess AFTER split
    X_train_p, X_test_p = preprocess_data(X_train_full, X_test)

    # run experiment on different T's
    T = [1, 3, 5, 10, 20, 50]
    results = []

    for t_val in T:
        model = BoostingClassifier(T=t_val)
        model.train(X_train_p, y_train_full)

        preds = model.predict(X_test_p)
        acc = evaluate(y_test, preds)

        results.append(acc)
        print(t_val, acc)
    
    plt.plot(T, results, marker='o')

    plt.xlabel("Number of boosting rounds T")
    plt.ylabel("Accuracy")
    plt.title("AdaBoost Accuracy vs Number of boosting rounds T")
    plt.grid(True)
    plt.savefig("adaboost_plot.png")
