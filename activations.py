# Activation functions
# 
#  Usage: import [pkg_path.]activations as act
#
#  Every activation function returns two numpy arrays, the first one is activation values,
#  second is the derivatives at that value.

import numpy as np

def linear(z, lbound = -np.Inf, ubound = np.Inf):
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
        np.ndarray -- a, da = a'
    """

    a = np.minimum(np.maximum(z, lbound), ubound) + 0.0
    da = ((z <= ubound) & (z >= lbound)) + 0.0
    return a, da


def identity(z):
    return linear(z)


def relu(z):
    return linear(z, lbound = 0.0)


def leaky_relu(z, slope = 0.01):
    """leaky ReLU
    
    a = max(slope * z, z)
    a' = 1 if z >= 0.0
       = slope if z < 0.0
    
    Arguments:
        z {np.ndarray} -- input values
    
    Keyword Arguments:
        slope {number} -- slope when z < 0.0, between 0,0 and 1.0 (default: {0.01})
    
    Returns:
        np.ndarray -- a, da = a'
    """

    a = np.maximum(z, slope * z) + 0.0
    da = (z >= 0) + (z < 0) * slope
    return a, da


def sigmoid(z):
    """sigmoid

    a = 1 / (1 + exp(-z))
    a' = a * (1-a)

    Arguments:
        z {np.ndarray} -- input values

    Returns:
        np.ndarray -- a, da = a'
    """

    a = 1.0 / (1 + np.exp(-z))
    da = a * (1.0 - a)

    return a, da


def tanh(z):
    """tanh

    a = tanh(z)
    a' = 1 - a ^ 2

    Arguments:
        z {np.ndarray} -- input values

    Returns:
        np.ndarray -- a, da = a'
    """

    a = np.tanh(z)
    da = 1.0 - a ** 2

    return a, da

