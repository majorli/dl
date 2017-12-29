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

def one_hot_encoding(Y):
    """one-hot encoding

    E.g. from [0, 1, 2] to [[1, 0, 0],[0, 1, 0],[0, 0, 1]]

    Arguments:
        Y -- Row vector with positive integer labels

    Returns:
        Y_oh -- Matrix of one-hot labels
    """
    C = np.max(Y) + 1
    Y_oh = np.zeros((C, Y.shape[1]))
    for c in range(C):
        Y_oh[c, :] = Y_oh[c, :] + (Y == c)

    return Y_oh.astype(np.int64)

def one_hot_decoding(Y_oh):
    """one-hot decoding

    Inverse of one-hot encoding.
    E.g. from [0, 1, 2] to [[1, 0, 0],[0, 1, 0],[0, 0, 1]]

    Arguments:
        Y_oh -- Matrix of one-hot labels

    Returns:
        Row vector with positive integer labels
    """
    return np.argmax(Y_oh, axis=0).reshape(1, -1).astype(np.int64)

def normalize(X, mean=None, stdev=None):
    """normalize the dataset
    
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
        mu = np.mean(X, axis=1, keepdims=True)

    if stdev is not None:
        sigma = stdev
    else:
        sigma = np.std(X, axis=1, keepdims=True)

    return (X - mu) / sigma, mu, sigma

def shuffle(X, Y=None):
    """shuffle the dataset
    
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

def raise_ord(X, ord=2):
    """raise order of the dataset"""
    n = X.shape[0]
    items = []
    nums = []

    for first in range(n):
        lst = [first for i in range(ord)]
        while lst[0] == first:
            s = True
            for k in range(ord-1, 0, -1):
                if lst[k] < lst[k-1]:
                    s = False
                    break
            if s:
                items.append(lst.copy())
            lst[ord-1] += 1
            for j in range(ord-1, 0, -1):
                if lst[j] >= n:
                    lst[j] = first
                    lst[j-1] += 1

    for l in range(len(items)):
        xl = np.ones(X.shape[1])
        for i in range(len(items[l])):
            xl = xl * X[items[l][i], :]
        nums.append(xl)

    return np.array(nums)

def polynomial(X, ord=2):
    """make the dataset to high order polynomial"""
    _X = X.copy()
    for d in range(2, ord + 1):
        _X = np.vstack((_X, raise_ord(X, d)))

    return _X

# ******************** #
# Activation functions #
# ******************** #

def softmax(X):
    """softmax
    
    The activation function of output layer in softmax regression N.N. to solve multiple classification problems.

    exp(X) / sum(exp(X))
    
    Arguments:
        X -- dataset
    
    Returns:
        softmax of X
    """
    assert(X is not None)
    T = np.exp(X)
    S = np.sum(T, axis = 0, keepdims = True)
    return T / S

def sigmoid(z):
    """sigmoid

    The activation function of output layer in logistic regression N.N. to solve classification problems.

    a = 1 / (1 + exp(-z))

    Arguments:
        z -- data or dataset

    Returns:
        sigmoid of z
    """
    assert(z is not None)
    return 1.0 / (1 + np.exp(-z))

def linear(z):
    """linear, just identity

    The activation function of output layer in linear regression N.N. to solve continuous value prediction problems.

    Arguments:
        z -- data or dataset

    Returns:
        identity to z
    """
    assert(z is not None)
    return z

def bounded_linear(z, lbound=-np.Inf, ubound=np.Inf):
    """Bounded linear function

    a = z if lbound < z < ubound;
      = lbound if z <= lbound;
      = ubound if z >= ubound
    a' = 1 if lbound <= z <= ubound;
       = 0 otherwise

    Arguments:
        z {np.ndarray} -- input values

    Keyword Arguments:
        lbound {number} -- lower bound (default: {-infinite})
        ubound {number} -- upper bound (default: {infinite})

    Returns:
        a -- linear(z)
        da -- derivative of linear(z) at point 'a'
    """
    a = np.minimum(np.maximum(z, lbound), ubound) + 0.0
    da = ((z <= ubound) & (z >= lbound)) + 0.0
    return a, da

def relu(z):
    """ReLU

    The most commonly used activation function for hidden units in N.N. to generate positive linear transform for values.

    Arguments:
        z -- input value

    Returns:
        a -- relu(z)
        da -- derivative of relu(z) at point 'a'
    """
    assert(z is not None)
    return linear(z, lbound=0.0)

def tanh(z):
    """tanh

    Commonly used activation function for hidden units in N.N. to generate nonlinear transfrom for values.

    a = tanh(z)
    a' = 1 - a ^ 2

    Arguments:
        z {np.ndarray} -- input values

    Returns:
        a -- tanh(z)
        da -- derivative of tanh(z) at point 'a'
    """
    assert(z is not None)
    a = np.tanh(z)
    da = 1.0 - a ** 2
    return a, da

def leaky_relu(z, slope=0.01):
    """leaky ReLU

    a = max(slope * z, z)
    a' = 1 if z >= 0.0
       = slope if z < 0.0

    Arguments:
        z {np.ndarray} -- input values

    Keyword Arguments:
        slope {number} -- slope when z < 0.0, between 0,0 and 1.0 (default: {0.01})

    Returns:
        a -- leaky_relu(z)
        da -- derivative of leaky_relu(z) at point 'a'
    """
    assert(z is not None)
    a = np.maximum(z, slope * z) + 0.0
    da = (z >= 0) + (z < 0) * slope
    return a, da

# ****************** #
# Dataset generators #
# ****************** #

def round_ds(dims=2, num=100, scale=20.0, rad=6.0, blur=1.0, alien=0.02):
    X = np.random.rand(dims, num) * scale
    N = np.linalg.norm(X - scale / 2, ord = 2, axis = 0, keepdims = True)
    Y = ((N < rad - blur) | ((N >= rad - blur) & (N <= rad + blur) & (np.random.rand(N.shape[0], N.shape[1]) < 0.5))) ^ (np.random.rand(N.shape[0], N.shape[1]) < alien)
    return X, Y + 0

def roundn_ds(dims=2, num=100, mu=0.0, stdev=1.0, rad=1.0, blur=0.2, alien=0.02):
    X = np.random.normal(mu, stdev, (dims, num))
    N = np.linalg.norm(X - mu, ord = 2, axis = 0, keepdims = True)
    Y = ((N < rad - blur) | ((N >= rad - blur) & (N <= rad + blur) & (np.random.rand(N.shape[0], N.shape[1]) < 0.5))) ^ (np.random.rand(N.shape[0], N.shape[1]) < alien)
    return X, Y + 0
