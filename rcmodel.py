import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import warnings

from rcio import *

class Model:
    """model"""
    _Y = None
    _Y_mean = None
    _Y_norm = None
    _pred_mask = None
    _test_mask = None

    _test_prop = 0.0
    _num_features = 0
    _algo = ""
    _learning_rate = 0.0
    _L2 = 0.0
    _steps_trained = 0

    _X = None
    _Theta = None

    def __init__(self, ref):
        if ref == "":
            # using default hyperparameters
            rc_state("Model with default hyperparameters, creating...")
            self._Y = None
            self._Y_mean = None
            self._Y_norm = None
            self._pred_mask = None
            self._test_mask = None

            self._test_prop = 0.1
            self._num_features = 100
            self._algo = "Adam"
            self._learning_rate = 0.01
            self._L2 = 0.0

            self._steps_trained = 0
            self._X = None
            self._Theta = None
        else:
            # load from a reference model early saved
            pass
        return

    def fed(self, Y):
        rc_state("Feeding training set...")
        self._Y = Y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._Y_mean = np.nan_to_num(np.nanmean(self._Y, axis=1, keepdims=True))
        self._Y_norm = self._Y - self._Y_mean
        self._pred_mask = self._Y - self._Y
        self._test_mask = None

        self._steps_traind = 0
        self._X = None
        self._Theta = None
        rc_result("Okay!")
        return

    def states(self):
        s = self._algo + ", prediction mask is "
        if self._pred_mask is None:
            s += "None, "
        else:
            s += "Okay, "
        s += str(self._num_features) + " features, learning_rate = "
        s += str(self._learning_rate) + ", L2 regularization = "
        s += str(self._L2) + ", "
        s += str(self._steps_trained) + " steps trained."

        return s
        
