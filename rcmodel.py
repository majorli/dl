import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import json

from rcio import *

def _results(X, Theta, Y_mean, relu=True):
    R = np.dot(X, Theta.T) + Y_mean
    if relu:
        R = np.maximum(R, 0.0)
    R = R + 0.0
    return R

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
    _eva = None

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
            self._num_features = 200
            self._algo = "Adam"
            self._learning_rate = 0.001
            self._L2 = 0.1
        else:
            # load from a reference model early saved
            npz = np.load("model_" + ref + ".npz")
            self._num_features = int(npz["nf"])
            self._algo = str(npz["algo"])
            self._learning_rate = float(npz["lr"])
            self._L2 = float(npz["L2"])
            pass

        self._steps_trained = 0
        self._X = None
        self._Theta = None
        self._eva = None
        return

    def run(self, _products, _customers):
        while True:
            rc_state(self.states())
            rc_highlight("With a running model, you can do these thing:")
            print("  1. Change algorithm.")
            print("  2. Try new learning rate.")
            print("  3. Change L2 regularization parameters.")
            print("  4. Change number of features.")
            print("  5. Optimize the model.")
            print("  6. Plot results.")
            print("  7. Look up results.")
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
                if self._algo == "Adam":
                    self._algo = "G.D."
                else:
                    self._algo = "Adam"
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
                y = rc_highlight_in("Enter new L2 parameter, remember it's a non-negative real number: ")
                try:
                    l = float(y)
                    if l < 0.0:
                        rc_fail("What a silly number you have entered! L2 parameter must be zero or positive.")
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
                if self._steps_trained > 0:
                    while True:
                        y = rc_warn_in("This model has been trained for " + str(self._steps_trained) + " steps. Would you like to reset it and learn from the very beginning? Choose NO to continue learning (Y/N)? ").upper()
                        if y != "" and (y[0] == "Y" or y[0] == "N"):
                            y = y[0]
                            break
                    if y == 'Y':
                        self._steps_trained = 0
                        self._X = None
                        self._Theta = None

                y = rc_highlight_in("How many steps would you like to learn (default: 1000)? ")
                try:
                    num_steps = int(y)
                    if num_steps <= 0:
                        num_steps = 1000
                except ValueError:
                    num_steps = 1000

                while True:
                    y = rc_highlight_in("Would you like to keep negative results (Y/N)? ").upper()
                    if y != "" and (y[0] == "Y" or y[0] == "N"):
                        y = y[0]
                        break
                relu = y == "N"             # whether to apply relu on the prediction or not
                rc_state("Optimizing...")
                self.optimize(num_steps)
                rc_state("Predicting...")
                R = _results(self._X, self._Theta, self._Y_mean, relu)
                rc_state("Evaluating...")
                # print bias, variance, precisions
                m_train = (~ self._pred_mask) & (~ self._test_mask)
                pred_train = R[m_train]
                real_train = self._Y[m_train]
                pred_test = R[self._test_mask]
                real_test = self._Y[self._test_mask]
                error_train = np.abs(pred_train - real_train)
                TE_train = np.sum(error_train)
                AE_train = TE_train / real_train.shape[0]
                WE_train = np.max(error_train)
                BE_train = np.min(error_train)
                Bias = TE_train / np.sum(np.abs(real_train))
                error_test = np.abs(pred_test - real_test)
                TE_test = np.sum(error_test)
                AE_test = TE_test / real_test.shape[0]
                WE_test = np.max(error_test)
                BE_test = np.min(error_test)
                Variance = TE_test / np.sum(np.abs(real_test))
                self._eva = {
                        "TE_train" : TE_train,
                        "AE_train" : AE_train,
                        "BE_train" : BE_train,
                        "WE_train" : WE_train,
                        "Bias" : Bias,
                        "TE_test" : TE_test,
                        "AE_test" : AE_test,
                        "BE_test" : BE_test,
                        "WE_test" : WE_test,
                        "Variance" : Variance,
                        }
                rc_result("1. Evaluating by " + str(pred_train.shape[0]) + " training examples:")
                rc_result("   TOTAL ERROR = " + str(TE_train))
                rc_result("   AVERAGE ERROR = " + str(AE_train))
                rc_result("   BEST ERROR = " + str(BE_train))
                rc_result("   WORST ERROR = " + str(WE_train))
                rc_warn("   BIAS = " + str(Bias))
                rc_result("2. Evaluating by " + str(pred_test.shape[0]) + " test examples:")
                rc_result("   TOTAL ERROR = " + str(TE_test))
                rc_result("   AVERAGE ERROR = " + str(AE_test))
                rc_result("   BEST ERROR = " + str(BE_test))
                rc_result("   WORST ERROR = " + str(WE_test))
                rc_warn("   VARIANCE = " + str(Variance))
                # export prediction results and precisions
                y = rc_highlight_in("Exporting results, give me a filename (without extname), nothing to not export: ")
                if y != "":
                    with open("results_" + y + ".csv", "w", newline="") as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow([" ", " ", " "] + self._axis_p)
                        f_csv.writerow([" ", " ", " "] + [_products[i] for i in self._axis_p])
                        f_csv.writerow([" ", " ", " "] + list(np.sum(R > 0.0, axis=1)))
                        count_by_cust = list(np.sum(R > 0.0, axis=0))
                        nc = len(self._axis_c)
                        for i in range(nc):
                            c = self._axis_c[i]
                            row = [c, _customers[c], count_by_cust[i]] + list(R[:, i])
                            f_csv.writerow(row)
                            pass
                        pass
                    f_json = open("results_eva_" + y + ".json", "w")
                    json.dump(self._eva, f_json)
                    f_json.close()
                    rc_state("Results exported.")
                else:
                    rc_warn("Results not exported.")
            elif opt == 6:
                # plot result
                fig = plt.figure()
                ax = fig.gca(projection="3d")
                X = np.arange(self._Y.shape[0])
                Y = np.arange(self._Y.shape[1])
                X, Y = np.meshgrid(X, Y)
                Z = self._Y[X, Y]
                surf = ax.plot_surface(X, Y, Z)
                ## plot prediction
                if self._X is not None and self._Theta is not None:
                    R = _results(self._X, self._Theta, self._Y_mean, True)
                    # x_test, y_test, z_test, x_pred, y_pred, z_pred = [], [], [], [], [], []
                    x_pred, y_pred, z_pred = [], [], []
                    for i in range(R.shape[0]):
                        for j in range(R.shape[1]):
#                            if self._test_mask[i, j]:
#                                x_test.append(i)
#                                y_test.append(j)
#                                z_test.append(R[i, j])
                            if self._pred_mask[i, j]:
                                x_pred.append(i)
                                y_pred.append(j)
                                z_pred.append(R[i, j])
#                    scat_test = ax.scatter(x_test, y_test, z_test, c="tab:orange", marker=".")
                    scat_pred = ax.scatter(x_pred, y_pred, z_pred, c="m", marker=".")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.show()
            elif opt == 7:
                # look up result
                # TODO
                pass
            elif opt == 8:
                # cluster
                # TODO
                pass
            elif opt == 9:
                # save model
                while True:
                    y = rc_highlight_in("Enter the name of this model: ")
                    if y != "":
                        break
                self.save(y)
                rc_result("Saved Okay as '" + y + "'.")
            else:
                pass

        # end while
        return

    def optimize(self, num_steps):
        # get dataset shape
        (num_products, num_customers) = self._Y.shape

        # constant: L2 regularization parameter
        l2 = tf.constant(self._L2)

        # placeholder: dataset to learn
        y = tf.placeholder(tf.float32, shape=[num_products, num_customers])

        # Variables: feature vectors for all customers (theta) and all products (x)
        if self._X is None or self._Theta is None or self._steps_trained == 0:
            x = tf.Variable(tf.random_normal([num_products, self._num_features]))
            theta = tf.Variable(tf.random_normal([num_customers, self._num_features]))
        else:
            x = tf.Variable(self._X, dtype=tf.float32)
            theta = tf.Variable(self._Theta, dtype=tf.float32)

        # computation graph
        pred = tf.matmul(x, tf.transpose(theta))
        diff = pred - y
        d = tf.where(tf.is_nan(diff), tf.zeros_like(diff), diff)

        # cost function
        cost = (tf.reduce_sum(tf.square(d)) + l2 * (tf.reduce_sum(tf.square(x)) + tf.reduce_sum(tf.square(theta)))) / 2.0

        # optimizer
        if self._algo == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(cost)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(cost)

        # variable initialzer
        init = tf.global_variables_initializer()

        # learn
        sess = tf.Session()
        sess.run(init)

        Ynorm = self._Y_norm.copy()
        if self._steps_trained == 0:
            self._test_mask = ~ self._pred_mask & (np.random.rand(num_products, num_customers) < 0.003)
        Ynorm[self._test_mask] = np.nan

        for t in range(num_steps):
            _, c = sess.run([optimizer, cost], feed_dict ={y:Ynorm})
            if t % 100 == 0:
                print("step: ", t, ", cost: ", c)

        # result
        self._X = sess.run(x)
        self._Theta = sess.run(theta)
        self._steps_trained += num_steps

        sess.close()

        return

    def save(self, fn):
        np.savez("model_" + fn, Y=self._Y, axisp=self._axis_p, axisc=self._axis_c, st=self._steps_trained, nf=self._num_features, X=self._X, Theta=self._Theta, algo=self._algo, lr=self._learning_rate, L2=self._L2)
        return

    def load(self, fn):
        rc_state("Loading model " + fn)
        npz = np.load("model_" + fn + ".npz")
        if npz["Y"].shape == ():
            self._Y = None
        else:
            self._Y = npz["Y"]
        if self._Y is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._Y_mean = np.nan_to_num(np.nanmean(self._Y, axis=1, keepdims=True))
            self._Y_norm = self._Y - self._Y_mean
            self._pred_mask = np.isnan(self._Y)     # self._Y - self._Y
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
        self._pred_mask = np.isnan(self._Y)     # self._Y - self._Y
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
        s += str(self._steps_trained) + " steps trained, "
        if self._eva is None:
            s += "last results: None."
        else:
            s += "last results: Bias = " + str(self._eva["Bias"]) + ", Variance = " + str(self._eva["Variance"]) + "."
        return s

    def not_ready(self):
        return self._Y is None
