#!/usr/bin/env python3
# Lauren Cho
# A2N1T01 

"""
CMPSC 165B - Machine Learning
Homework 3, Problem 1: K-Nearest Neighbors
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


class KNNClassifier:
    """K-Nearest Neighbors Classifier."""
    # initialize our stuff
    def __init__(self, k=1):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """Fit the classifier to training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict labels for input samples."""
        predictions = []

        # Compute the distance from x to every training point
        for x in X:
            distances = []

            for train_pt, label in zip(self.X_train, self.y_train):
                dist = np.sqrt(np.sum((x - train_pt) ** 2))
                distances.append((dist, label))

            # Select the k nearest training points
            distances.sort(key=lambda pair: (pair[0], pair[1]))
            neighbors = distances[:self.k]

            # count votes
            count_pos = 0
            count_neg = 0

            for n in neighbors:
                # Predict the most common label among these k neighbors
                if (n[1] ==1):
                    count_pos += 1
                elif (n[1] == -1):
                    count_neg += 1

            # Predict with tie-breaking
            if count_pos > count_neg:
                predictions.append(1)
            elif count_neg > count_pos:
                predictions.append(-1)
            else:
                # tie in votes: choose label of closest neighbor
                predictions.append(neighbors[0][1])

        return np.array(predictions)


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
    model = KNNClassifier(k=2)  # tune k
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

    # run experiment on different k's
    ks = [1, 3, 5, 7, 9, 11, 15, 21]
    results = []

    for k_val in ks:
        model = KNNClassifier(k=k_val)
        model.train(X_train_p, y_train_full)

        preds = model.predict(X_test_p)
        acc = evaluate(y_test, preds)

        results.append(acc)
        print(k_val, acc)
    
    plt.plot(ks, results, marker='o')

    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN Classifer Accuracy vs k Used")
    plt.grid(True)
    plt.savefig("knn_classifer_plot.png")
