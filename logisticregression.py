import numpy as np
import  pandas as pd
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, solver='liblinear'):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.solver = solver
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def gradient_descent(self, X, y, theta, learning_rate, iterations):
        m = len(y)
        for _ in range(iterations):
            h = self.sigmoid(np.dot(X, theta))
            gradient = np.dot(X.T, (h - y)) / m
            theta -= learning_rate * gradient
        return theta

    def normalize_feature(self, feature):
        return (feature - np.mean(feature)) / np.std(feature)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize features if needed
        X[:, :3] = self.normalize_feature(X[:, :3])

        # Add intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # Initialize parameters
        self.theta = np.zeros(X.shape[1])

        # Train the model using gradient descent
        self.theta = self.gradient_descent(X, y, self.theta, self.learning_rate, self.iterations)

        return self

    def predict_proba(self, X, mean=None, std=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

    # Normalize features using mean and std from training set
        if mean is not None and std is not None:
            X[:, :1] = (X[:, :1] - mean) / std

    # Add intercept term
        X = np.c_[np.ones(X.shape[0]), X]

    # Predict probabilities
        probabilities = self.sigmoid(np.dot(X, self.theta))
        return np.column_stack((1 - probabilities, probabilities))

      
  

    def predict(self, X, threshold=0):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
    
    def calculate_probabilities(self, X):
            # Calculate winning probabilities
        winning_probabilities = self.predict_proba(X)[:, 1]

        # Calculate losing probabilities
        losing_probabilities = 1 - winning_probabilities

        return winning_probabilities, losing_probabilities
