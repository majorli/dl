# Comvolution Neural Network model

import numpy as np

def pad(X, padding=1, value=0.0, dims=(1, 2)):
    """padding channels

        Padding channels

    Arguments:
        X -- Dataset

    Keyword Arguments:
        padding -- padding border (default: {1})
        value -- padding value (default: {0.0})
        dims -- padding dimensions (default: {(1, 2)})

    Return:
        X_pad -- Padded dataset
    """
    pd = [(0, 0) for d in range(len(X.shape))]
    for d in dims:
        pd[d] = (padding, padding)
    X_pad = np.pad(X, tuple(pd), "constant", constant_values=(value, value))
    return X_pad

