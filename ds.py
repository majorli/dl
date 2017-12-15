# Functions for manipulations of dataset
#
#  Dataset for machine learning is a matrix composed of a set of feature
#  vectors of examples.
#  Conventionally, we denote the number of examples in a dataset as 'm'
#  and the number of features of example as 'n_x'.
#  Therefore, a dataset 'X' is a (n_x * m) matrix.
#  If the dataset is a labeled dataset used for supervised learning, then
#  there should be a label vector 'Y' which is a row vector contains 'm'
#  elements. Y[i] is the real result of example X[:,i]

import numpy as np


def normalize(X, mean = None, stdev = None):
    """
    
    Normalize features
    
    Arguments:
        X {np.ndarray} -- dataset
    
    Keyword Arguments:
        mean {number} -- mean used to zero out data, None = use mean of given data, (default: {None})
        stdev {number} -- standard deviation to normalize data, None = use stdev of given data (default: {None})
    
    Returns:
        np.ndarray, number, number -- normalized data, mean, stdev
    """
    if mean is not None:
        mu = mean
    else:
        mu = np.mean(X, axis = 1, keepdims = True)
    X = X - mu

    if stdev is not None:
        sigma = stdev
    else:
        sigma = np.std(X, axis = 1, keepdims = True)

    X = X / sigma

    return X, mu, sigma


def softmax(X):
    """
    
    Softmax features
    X[i,j] := exp(X[i,j]) / sum_k=1..m(exp(X[i,k]))
    
    Arguments:
        X {np.ndarray} -- dataset
    
    Returns:
        np.ndarray -- softmaximized dataset
    """

    _X = X.astype(float)
    X_exp = np.exp(_X)
    X_sum = np.sum(X_exp, axis=1, keepdims=True)
    _X = X_exp / X_sum

    return _X


def shuffle(X):
    """
    
    Shuffle examples in dataset
    
    Arguments:
        X {np.ndarray} -- dataset
    
    Returns:
        np.ndarray -- shuffled dataset
    """
    
    return np.random.permutation(X.T).T
