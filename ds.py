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

# ***************** #
# Dataset Utilities #
# ***************** #

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
    X_exp = np.exp(X)
    X_sum = np.sum(X_exp, axis = 1, keepdims = True)
    X = X_exp / X_sum

    return X

def shuffle(X, Y = None):
    """
    
    Shuffle examples in dataset
    
    Arguments:
        X {np.ndarray} -- dataset
    
    Returns:
        np.ndarray -- shuffled dataset
    """
    permutation = np.random.permutation(X.shape[1])
    X_shuffled = X[:, permutation]
    if Y is not None and Y.shape[1] == X.shape[1]:
        Y_shuffled = Y[:, permutation]
    else:
        Y_shuffled = Y
    return X_shuffled, Y_shuffled

# ****************** #
# Dataset generators #
# ****************** #

def round_ds(dims = 2, num = 100, scale = 20.0, rad = 6.0, blur = 1.0, alien = 0.02):
    X = np.random.rand(dims, num) * scale
    N = np.linalg.norm(X - scale / 2, ord = 2, axis = 0, keepdims = True)
    Y = ((N < rad - blur) | ((N >= rad - blur) & (N <= rad + blur) & (np.random.rand(N.shape[0], N.shape[1]) < 0.5))) ^ (np.random.rand(N.shape[0], N.shape[1]) < alien)
    return X, Y + 0

def roundn_ds(dims = 2, num = 100, mu = 0.0, stdev = 1.0, rad = 1.0, blur = 0.2, alien = 0.02):
    X = np.random.normal(mu, stdev, (dims, num))
    N = np.linalg.norm(X - mu, ord = 2, axis = 0, keepdims = True)
    Y = ((N < rad - blur) | ((N >= rad - blur) & (N <= rad + blur) & (np.random.rand(N.shape[0], N.shape[1]) < 0.5))) ^ (np.random.rand(N.shape[0], N.shape[1]) < alien)
    return X, Y + 0
