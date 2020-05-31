# This is the code used to create the MNIST data files as .csv files
from sklearn.datasets import fetch_openml
import pandas as pd
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# Normalization of values
X = X / 255.
# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
pd.DataFrame(X_train).to_csv("X_train.csv", header=None, index=None)
pd.DataFrame(X_test).to_csv("X_test.csv", header=None, index=None)
pd.DataFrame(y_train).to_csv("y_train.csv", header=None, index=None)
pd.DataFrame(y_test).to_csv("y_test.csv", header=None, index=None)