#!/usr/bin/env python3
# Lauren Cho
# A2N1T01 

"""
CMPSC 165B - Machine Learning
Homework 3, Problem 2: Boosting Classifier
"""

import numpy as np
import pandas as pd


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

    X_train_new = np.hstack([X_train_new, np.ones((X_train_new.shape[0], 1), dtype=X_train_new.dtype)])
    X_test_new = np.hstack([X_test_new, np.ones((X_test_new.shape[0], 1), dtype=X_test_new.dtype)])

    if np.isnan(X_train_new).any() or np.isnan(X_test_new).any():
        raise ValueError("NaNs found after preprocessing — data loading/conversion is wrong.")
  
    return X_train_new, X_test_new


class BoostingClassifier:
    """AdaBoost Classifier with weighted linear classifier as weak learner."""
    # initialize our stuff
    def __init__(self):
      

    def train(self, X, y):
        """Fit the classifier to training data."""
        # TODO: Implement
        raise NotImplementedError

    def predict(self, X):
        """Predict labels for input samples."""
        # TODO: Implement
        raise NotImplementedError


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
    model = VotedPerceptron(epochs=3)  # tune epochs
    model.train(X_train_p, y_train)

    # Predict
    y_pred = model.predict(X_test_p).astype(int)

    # Save one integer per line, no header according to assignment instructions
    np.savetxt(pred_file, y_pred, fmt="%d")
