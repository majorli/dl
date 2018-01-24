import numpy as np

import dl.kmeans as km
import dl.recommender as rc
from rcio import *

class Model:
    """model"""
    _pred_mask = None
    _test_mask = None
    _num_features = 0
    _algo = ""
    _learning_rate = 0.0
    _L2 = 0.0
    _steps_trained = 0

    def __init__(self, ref):
        if ref == "":
            # using default hyperparameters
            rc_state("Model with default hyperparameters, creating...")
            self._pred_mask = None
            self._test_mask = None
            self._num_features = 100
            self._algo = "Adam"
            self._learning_rate = 0.01
            self._L2 = 0.0
            self._steps_trained = 0
        else:
            pass

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
        
