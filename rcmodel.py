import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from rcio import *

class Model:
    """model"""
    _Y = None
    _Y_mean = None
    _Y_norm = None
    _pred_mask = None
    _test_mask = None
    _axis_p = None
    _axis_c = None

    _test_prop = 0.0
    _num_features = 0
    _algo = ""
    _learning_rate = 0.0
    _L2 = 0.0
    _steps_trained = 0

    _X = None
    _Theta = None

    def __init__(self, ref):
        rc_state("Model creating...")
        self._Y = None
        self._Y_mean = None
        self._Y_norm = None
        self._pred_mask = None
        self._test_mask = None
        self._axis_p = None
        self._axis_c = None
        self._test_prop = 0.0
        if ref == "":
            # using default hyperparameters
            self._num_features = 100
            self._algo = "adam"
            self._learning_rate = 0.01
            self._L2 = 0.0
        else:
            # load from a reference model early saved
            npz = np.load(ref + ".npz")
            self._num_features = int(npz["nf"])
            self._algo = str(npz["algo"])
            self._learning_rate = float(npz["lr"])
            self._L2 = float(npz["L2"])
            pass

        self._steps_trained = 0
        self._X = None
        self._Theta = None
        return

    def run(self):
        while True:
            rc_state(self.states())
            rc_highlight("With a running model, you can do these thing:")
            print("  1. Change algorithm.")
            print("  2. Try new learning rate.")
            print("  3. Change L2 regularization parameters.")
            print("  4. Change number of features.")
            print("  5. Optimize the model.")
            print("  6. Plot result.")
            print("  7. Export results.")
            print("  8. Cluster products and customers.")
            print("  9. Save model.")
            opt = 0
            while True:
                o = rc_highlight_in("What would you like (1..9, 0 to exit): ")[0]
                if o.isdigit():
                    opt = int(o)
                    break
            # end while
            if opt == 0:
                break
            if opt == 1:
                # change algorithm
                if self._algo == "adam":
                    self._algo = "G.D."
                else:
                    self._algo = "adam"
                # set steps trained to zero, must optimize from the begin and can't do 6, 7, 8 now
                self._steps_trained = 0
                rc_result("Model algorithm is changed to " + self._algo + " and model has reset.")
            elif opt == 2:
                # change learning rate
                y = rc_highlight_in("Enter new learning rate, remember it's a small positive real number: ")
                try:
                    l = float(y)
                    if l <= 0.0:
                        rc_fail("What a silly number you have entered! The learning rate must be positive.")
                        rc_warn("Incorrect value, nothing changed.")
                    else:
                        self._learning_rate = l
                        rc_result("Learning rate is changed to " + str(l) + ".")
                except ValueError:
                    rc_fail("Haven't I told you that learning rate must be a number?")
                    rc_warn("Incorrect value, nothing changed.")
            elif opt == 3:
                # change L2
                y = rc_highlight_in("Enter new L2 parameter, remember it's a positive real number: ")
                try:
                    l = float(y)
                    if l <= 0.0:
                        rc_fail("What a silly number you have entered! L2 parameter must be positive.")
                        rc_warn("Incorrect value, nothing changed.")
                    else:
                        self._L2 = l
                        rc_result("L2 regularization parameter is changed to " + str(l) + ".")
                except ValueError:
                    rc_fail("Haven't I told you that L2 parameter must be a number?")
                    rc_warn("Incorrect value, nothing changed.")
            elif opt == 4:
                # change number of features
                y = rc_highlight_in("Enter the number of features, a positive integer please: ")
                try:
                    l = int(y)
                    if l <= 0:
                        rc_fail("What a silly number you have entered! The number of features must be positive.")
                        rc_warn("Incorrect value, nothing changed.")
                    else:
                        self._num_features = l
                        # set steps trained to zero, must optimize from the begin and can't do 6, 7, 8 now
                        self._steps_trained = 0
                        rc_result("Now there will be " + str(l) + " features for each product and customer. Model has rest.")
                except ValueError:
                    rc_fail("I told you that number of features must be positive!")
                    rc_warn("Incorrect value, nothing changed.")
            elif opt == 5:
                # optimize
                rc_state("Optimizing...")
                self.optimize()
                rc_state("Completed!")
            elif opt == 6:
                # plot result
                fig = plt.figure()
                ax = fig.gca(projection="3d")
                X = np.arange(self._Y.shape[0])
                Y = np.arange(self._Y.shape[1])
                X, Y = np.meshgrid(X, Y)
                Z = self._Y[X, Y]
                surf = ax.plot_surface(X, Y, Z)
                ## TODO: plot scatters of results in masks, c='tab:orange' and 'm'
                ## s = ax.scatter(x, y, z, c='m', marker='.')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.show()
            elif opt == 7:
                # export result
                pass
            elif opt == 8:
                # cluster
                pass
            elif opt == 9:
                # save model
                while True:
                    y = rc_highlight_in("Enter the name of this model: ")
                    if len(y) > 0:
                        break
                self.save(y)
                rc_result("Saved Okay as '" + y + "'.")
            else:
                pass

        # end while
        return

    def optimize(self):
        # TODO optimize
        return

    def save(self, fn):
        np.savez(fn, Y=self._Y, axisp=self._axis_p, axisc=self._axis_c, st=self._steps_trained, nf=self._num_features, X=self._X, Theta=self._Theta, algo=self._algo, lr=self._learning_rate, L2=self._L2)
        return

    def load(self, fn):
        rc_state("Loading model " + fn)
        npz = np.load(fn + ".npz")
        if npz["Y"].shape == ():
            self._Y = None
        else:
            self._Y = npz["Y"]
        if self._Y is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._Y_mean = np.nan_to_num(np.nanmean(self._Y, axis=1, keepdims=True))
            self._Y_norm = self._Y - self._Y_mean
            self._pred_mask = self._Y - self._Y
            self._test_mask = np.zeros(self._Y.shape)
            self._axis_p = list(npz["axisp"])
            self._axis_c = list(npz["axisc"])
        else:
            self._Y_mean = None
            self._Y_norm = None
            self._pred_mask = None
            self._test_mask = None
            self._axis_p = None
            self._axis_c = None
        
        self._steps_trained = int(npz["st"])
        if npz["X"].shape == ():
            self._X = None
        else:
            self._X = npz["X"]
        if npz["Theta"].shape == ():
            self._Theta = None
        else:
            self._Theta = npz["Theta"]

        self._test_prop = 0.0
        self._num_features = int(npz["nf"])
        self._algo = str(npz["algo"])
        self._learning_rate = float(npz["lr"])
        self._L2 = float(npz["L2"])
        return

    def fed(self, ts):
        rc_state("Feeding training set...")
        self._Y = ts["Y"]
        self._axis_p = ts["P"]
        self._axis_c = ts["C"]
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
        if self._Y is None:
            s += "no training set fed, "
        else:
            s += str(len(self._axis_p)) + " products, "
            s += str(len(self._axis_c)) + " customers, "
        s += str(self._num_features) + " features, learning_rate = "
        s += str(self._learning_rate) + ", L2 regularization = "
        s += str(self._L2) + ", "
        s += str(self._steps_trained) + " steps trained."
        return s

    def not_ready(self):
        return self._Y is None
