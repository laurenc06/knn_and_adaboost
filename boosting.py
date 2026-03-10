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
    # TODO: Implement
    raise NotImplementedError


def preprocess_data(X_train, X_test):
    """Preprocess training and test data."""
    # TODO: Implement
    raise NotImplementedError


class BoostingClassifier:
    """AdaBoost Classifier with weighted linear classifier as weak learner."""

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
    # TODO: Implement
    raise NotImplementedError


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""
    # TODO: Implement
    raise NotImplementedError
