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


def normalize(X, stdev=False, centerize=False):
    """
    
    Normalize features
    
    Arguments:
        X {np.ndarray} -- dataset
    
    Keyword Arguments:
        stdev {bool} -- do normalization by using standard deviation (default: {False})
        centerize {bool} -- minus the mean of data to centered to zero (default: {False})
    
    Returns:
        np.ndarray -- normalized dataset
    """
    
    _X = X.astype(float)

    if centerize == True:
        X_mean = np.mean(_X, axis=1, keepdims=True)
        _X -= X_mean

    X_norm = None
    if stdev == True:
        X_norm = np.std(_X, axis=1, keepdims=True)
    else:
        X_norm = np.linalg.norm(_X, axis=1, ord=2, keepdims=True)
    print(str(X_norm))
    _X /= X_norm

    return _X


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
