# Functions for manipulations of dataset
#
#  Dataset for machine learning is a matrix composed of a set of feature vectors of examples.
#  Conventionally, we denote the number of examples in a dataset as 'm' and the number of features of example as 'n_x'.
#  Therefore, a dataset 'X' is a (n_x * m) matrix.

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
        a -- bounded_linear(z)
        da -- derivative of bounded_linear(z) at point 'a'
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
    a = np.maximum(z, 0.0) + 0.0
    da = (z >= 0) + 0.0
    return a, da

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

def rect_ds(num, mu=[0.0, 0.0], stdev=[1.0, 1.0], length=[1.0, 1.0], alien=0.005):
    assert(len(mu) == len(stdev))
    assert(len(mu) == len(length))
    d = len(mu)
    m = np.array(mu).reshape(d, 1)
    s = np.array(stdev).reshape(d, 1)
    r = np.array(length).reshape(d, 1)
    X = np.random.normal(m, s, (d, num))
    N = np.abs(X - m) < r
    Y = np.all(N, axis=0, keepdims=True) ^ (np.random.rand(1, N.shape[1]) < alien)
    return X, Y + 0

def round_ds(num, mu=[0.0, 0.0], stdev=[1.0, 1.0], radius=[1.0, 1.0], alien=0.005):
    assert(len(mu) == len(stdev))
    assert(len(mu) == len(radius))
    d = len(mu)
    m = np.array(mu).reshape(d, 1)
    s = np.array(stdev).reshape(d, 1)
    r = np.array(radius).reshape(d, 1)
    X = np.random.normal(m, s, (d, num))
    N = np.linalg.norm((X - m) / r, axis=0, keepdims=True) < 1.0
    Y = N ^ (np.random.rand(1, N.shape[1]) < alien)
    return X, Y + 0

def circle_ds(num, mu=[0.0, 0.0], stdev=[1.0, 1.0], radius=[1.0, 1.0], thickness=0.2, alien=0.005):
    assert(len(mu) == len(stdev))
    assert(len(mu) == len(radius))
    d = len(mu)
    m = np.array(mu).reshape(d, 1)
    s = np.array(stdev).reshape(d, 1)
    r = np.array(radius).reshape(d, 1)
    X = np.random.normal(m, s, (d, num))
    N = np.linalg.norm((X - m) / r, axis=0, keepdims=True)
    Y = ((N < 1.0 + thickness) & (N > 1.0 - thickness)) ^ (np.random.rand(1, N.shape[1]) < alien)
    return X, Y + 0

def multi_class_ds(num, centroids=[[0.0, 0.0], [2.0, 2.0]], radius=[[0.5, 0.5], [0.5, 0.5]]):
    c = np.array(centroids)
    r = np.array(radius)
    assert(c.shape == r.shape)
    K = c.shape[0]
    d = c.shape[1]
    assert(len(num) == K)
    X = np.zeros((d, 0))
    Y = np.zeros((K, 0))
    for k in range(K):
        Xk, _ = round_ds(num[k], c[k, :], r[k, :] * 0.75, r[k, :])
        X = np.hstack((X, Xk))
        Yk = np.zeros((K, num[k]))
        Yk[k, :] = np.ones((1, num[k]))
        Y = np.hstack((Y, Yk))
    return X, Y.astype(int)

# ****************** #
# Evaluating metrics #
# ****************** #

def logistic_metrics(pred, labels):
    """Metrics for logistic regression

    Accuracy, Precision, Recall and F1 score.

    Arguments:
        pred -- Predictions output by model
        lables -- Labels corresponding to predictions

    Returns:
        metrics -- Dictionary of evaluating metrics
    """
    assert(pred.shape == labels.shape)

    m = pred.shape[1]
    metrics = {}

    tp = np.sum((labels == 1) & (pred == 1))
    tn = np.sum((labels == 0) & (pred == 0))
    metrics["accuracy"] = (tp + tn) / m
    metrics["precision"] = tp / np.sum(pred)
    metrics["recall"] = tp / np.sum(labels)
    if tp == 0:
        metrics["F1"] = np.nan
    else:
        metrics["F1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])

    return metrics

def linear_metrics(pred, labels):
    """Metrics for linear regression

    Average, Max and Standard deviation of absolute error, relative error and distance between vectors in pred and labels.

    Arguments:
        pred -- Predictions output by model
        lables -- Labels corresponding to predictions

    Returns:
        metrics -- Dictionary of evaluating metrics
    """
    assert(pred.shape == labels.shape)

    d = pred.shape[0]
    m = pred.shape[1]
    metrics = {}

    e = np.abs(pred - labels)
    n = np.linalg.norm(e, axis=0, keepdims=True)
    a = e / labels
    metrics["absolute_errors"] = {
            "maximum" : np.max(e, axis=1, keepdims=True),
            "average" : np.mean(e, axis=1, keepdims=True),
            "stdev" : np.std(e, axis=1, keepdims=True)
            }
    metrics["relative_errors"] = {
        "maximum" : np.max(a, axis=1, keepdims=True),
        "average" : np.mean(a, axis=1, keepdims=True),
        "stdev" : np.std(a, axis=1, keepdims=True)
        }
    metrics["distance"] = {
        "maximum" : np.max(n),
        "average" : np.mean(n),
        "stdev" : np.std(n)
        }

    return metrics

def softmax_metrics(pred, labels):
    """Metrics for softmax regression

    Logistic regression metrics for each class.
    Accuracy of overall classes

    Arguments:
        pred -- Predictions output by model in one-hot encoding form
        lables -- Labels corresponding to predictions in one-hot encoding form

    Returns:
        metrics -- Dictionary of evaluating metrics
    """
    assert(pred.shape == labels.shape)
    assert(pred.shape[0] > 1)

    K = pred.shape[0]
    m = pred.shape[1]
    metrics = {}

    for k in range(K):
        metrics["class_" + str(k) + "_metrics"] = logistic_metrics(pred[k, :].reshape(1, m), labels[k, :].reshape(1, m))

    p = one_hot_decoding(pred)
    l = one_hot_decoding(labels)
    metrics["accuracy"] = np.sum(p == l) / m

    return metrics

