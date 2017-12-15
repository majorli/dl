# Evaluators

import numpy as np

def precision(Y, Predict, hypo_type = 1):
    """compute precision
    
    For regressions, compute average Frobenius norm; For classification problems, compute precision.
    
    Arguments:
        Y {np.ndarray} -- labels matrix
        Predict {np.ndarray} -- predicts matrix
    
    Keyword Arguments:
        hypo_type {number} -- 1 = classification; 2 = regression (default: {1})

    Returns:
        np.ndarray -- precision
    """
    assert(Y.shape == Predict.shape)
    m = Y.shape[1]

    if hypo_type == 1:
        Yhat = (Predict >= 0.5) + 0.0
        prec = np.sum((Y == Yhat), axis = 1, keepdims = True) / m
    else:
        Yhat = Predict - Y
        prec = np.linalg.norm(Yhat, axis = 1, keepdims = True) / m
    return prec