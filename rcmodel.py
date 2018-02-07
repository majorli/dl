import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
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
            npz = np.load("data/model_" + ref + ".npz")
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

    def _export_clusters(self, labels_p, labels_c, _products, _customers):

        csvfn = rc_getstr("Give me a name to export the clustering results, nothing to quit exporting: ", t="highlight", keepblank=True)
        if csvfn == "":
            rc_warn("Quit.")
            return

        n_p = len(np.unique(labels_p))
        n_c = len(np.unique(labels_c))
        cls = []
        for i in range(len(labels_p)):
            cl = labels_p[i]
            sid = self._axis_p[i]
            name = _products[sid]
            cls.append([cl, sid, name])
        cls.sort(key=lambda x: x[0])
        with open("results/clusters_products_" + csvfn + "_" + str(n_p) + "_cls.csv", "w", newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(["Cluster", "ProductId", "ProductName"])
            for it in cls:
                f_csv.writerow(it)

        cls = []
        for i in range(len(labels_c)):
            cl = labels_c[i]
            sid = self._axis_c[i]
            name = _customers[sid]
            cls.append([cl, sid, name])
        cls.sort(key=lambda x: x[0])
        with open("results/clusters_customers_" + csvfn + "_" + str(n_c) + "_cls.csv", "w", newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(["Cluster", "CustomerSId", "CustomerName"])
            for it in cls:
                f_csv.writerow(it)

        rc_result("Done! Clustering results exported.")
        return

    def _cluster_mean_shift(self):
        print("Trying products clustering...", end="", flush=True)
        ms_p = MeanShift(bandwidth=estimate_bandwidth(self._X)).fit(self._X)
        print("Done.", ms_p.cluster_centers_.shape[0], "clusters found.")

        print("Trying customers clustering...", end="", flush=True)
        ms_c = MeanShift(bandwidth=estimate_bandwidth(self._Theta)).fit(self._Theta)
        print("Done.", ms_c.cluster_centers_.shape[0], "clusters found.")

        return ms_p.labels_, ms_c.labels_

    def _cluster_affinity(self):
        pf_p = rc_getnum("Enter a non-positive number as the preference of products clustering, or nothing to use default value: ", t="highlight", blankas=None)
        if pf_p is not None and pf_p > 0.0:
            rc_warn("Incorrect! Should be a non-positive number. Use default parameter.")
            pf_p = None
        pf_c = rc_getnum("Enter a non-positive number as the preference of customers clustering, or nothing to use default value: ", t="highlight", blankas=None)
        if pf_c is not None and pf_c > 0.0:
            rc_warn("Incorrect! Should be a non-positive number. Use default parameter.")
            pf_c = None

        print("Trying products clustering...", end="", flush=True)
        if pf_p is None:
            af_p = AffinityPropagation().fit(self._X)
        else:
            af_p = AffinityPropagation(preference=pf_p).fit(self._X)
        print("Done.", af_p.cluster_centers_.shape[0], "clusters found.")
        print("Trying customers clustering...", end="", flush=True)
        if pf_c is None:
            af_c = AffinityPropagation().fit(self._Theta)
        else:
            af_c = AffinityPropagation(preference=pf_c).fit(self._Theta)
        print("Done.", af_c.cluster_centers_.shape[0], "clusters found.")

        return af_p.labels_, af_c.labels_

    def _cluster_kmeans(self):
        max_nc_p = rc_select("At most how many clusters do you think for products (2.." + str(self._Y.shape[0]-1) + ")? ", range_=range(2, self._Y.shape[0]), t="highlight")
        max_nc_c = rc_select("At most how many clusters do you think for customers (2.." + str(self._Y.shape[1]-1) + ")? ", range_=range(2, self._Y.shape[1]), t="highlight")

        n_p = []
        n_c = []
        dis_p = []
        dis_c = []
        lbl_p = []
        lbl_c = []
        for n in range(2, max_nc_p + 1):
            print("Trying products for", n, "clusters...", end="", flush=True)
            km = KMeans(n_clusters=n, n_init=10, max_iter=300, tol=1e-4).fit(self._X)
            print("Done. Distortion =", km.inertia_)
            n_p.append(n)
            dis_p.append(km.inertia_)
            lbl_p.append(km.labels_)
        for n in range(2, max_nc_c + 1):
            print("Trying customers for", n, "clusters...", end="", flush=True)
            km = KMeans(n_clusters=n, n_init=10, max_iter=300, tol=1e-4).fit(self._Theta)
            print("Done. Distortion =", km.inertia_)
            n_c.append(n)
            dis_c.append(km.inertia_)
            lbl_c.append(km.labels_)

        rc_result("All Done! Now I will show you the elbow plots, choose your number of clusters from the plots.")
        rc_warn("Remember: Not the lowest distortion is the best clustering, but the elbow point it is.")
        rc_state("Close the plots then you can choose clustering results. Feel free to save the plots.")
        fig = plt.figure("Clustering")
        ax = plt.subplot(211)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        _ = plt.plot(n_p, dis_p, ".-", label="Products Clustering")
        _ = plt.ylabel("Distortion")
        _ = plt.legend()
        ax = plt.subplot(212)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        _ = plt.plot(n_c, dis_c, ".-", label="Customers Clustering")
        _ = plt.xlabel("Number of clusters")
        _ = plt.ylabel("Distortion")
        _ = plt.legend()
        plt.show()

        n_p = rc_select("Enter the number of clusters for the best clustering of products you think of (2.." + str(max_nc_p) + "): ", range_=range(2, max_nc_p+1), t="highlight")
        n_c = rc_select("Enter the number of clusters for the best clustering of customers you think of (2.." + str(max_nc_c) + "): ", range_=range(2, max_nc_c+1), t="highlight")
        return lbl_p[n_p - 2], lbl_c[n_c - 2]

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
            print("  7. Cluster products and customers.")
            print("  8. Save model.")
            print("  0. Exit model running.")
            opt = rc_select("What would you like (0..8): ", range_=range(9), t="highlight")
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
                l = rc_getnum("Enter new learning rate, a small positive real number (Default: unchange): ", t="highlight", blankas=self._learning_rate)
                if l <= 0.0:
                    rc_fail("Havn't I told you that the learning rate must be positive.")
                    rc_warn("Incorrect value, nothing changed.")
                else:
                    self._learning_rate = l
                    rc_result("Learning rate is changed to {0}.".format(l))
            elif opt == 3:
                # change L2
                l = rc_getnum("Enter new L2 parameter, a non-negative real number (Default: unchange): ", t="highlight", blankas=self._L2)
                if l < 0.0:
                    rc_fail("Haven't I told you that L2 parameter must be a number?")
                    rc_warn("Incorrect value, nothing changed.")
                else:
                    self._L2 = l
                    rc_result("L2 regularization parameter is changed to {0}.".format(l))
            elif opt == 4:
                # change number of features
                l = rc_getint("Enter the number of features, a positive integer (Default: unchange): ", t="highlight", blankas=self._num_features)
                if l <= 0:
                    rc_fail("What a silly number you have entered! The number of features must be positive.")
                    rc_warn("Incorrect value, nothing changed.")
                else:
                    self._num_features = l
                    # set steps trained to zero, must optimize from the begin and can't do 6, 7, 8 now
                    self._steps_trained = 0
                    rc_result("Now there will be {0} features for each product and customer. Model has rest.".format(l))
            elif opt == 5:
                # optimize
                if self._algo == "Adam" or (self._steps_trained > 0 and rc_ensure("This is a trained G.D. model. Do you want to restart learning? (Y/N)? ", t="warn")):
                    self._steps_trained = 0
                    self._X = None
                    self._Theta = None

                num_steps = rc_getint("How many steps would you like to learn (default: 1000)? ", t="highlight", blankas=1000)
                if num_steps <= 0:
                    num_steps = 1000

                relu = not rc_ensure("Would you like to keep negative results (Y/N)? ", t="highlight")
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
                rc_result("1. Evaluating by {0} training examples:".format(pred_train.shape[0]))
                rc_result("   TOTAL ERROR = {0}".format(TE_train))
                rc_result("   AVERAGE ERROR = {0}".format(AE_train))
                rc_result("   BEST ERROR = {0}".format(BE_train))
                rc_result("   WORST ERROR = {0}".format(WE_train))
                rc_warn("   BIAS = {0}".format(Bias))
                rc_result("2. Evaluating by {0} test examples:".format(pred_test.shape[0]))
                rc_result("   TOTAL ERROR = {0}".format(TE_test))
                rc_result("   AVERAGE ERROR = {0}".format(AE_test))
                rc_result("   BEST ERROR = {0}".format(BE_test))
                rc_result("   WORST ERROR = {0}".format(WE_test))
                rc_warn("   VARIANCE = {0}".format(Variance))
                # export prediction results and precisions
                y = rc_getstr("Exporting results, give me a filename (without extname), nothing to not export: ", t="highlight", keepblank=True)
                if y != "":
                    with open("results/results_tbl_" + y + ".csv", "w", newline="") as f:
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
                    with open("results/results_pre_" + y + ".csv", "w", newline="") as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(["ProductId", "Product", "CustomerSid", "Customer", "SalesPrediction"])
                        mask_count = np.sum(self._pred_mask, axis=1)
                        (nP, nC) = R.shape
                        for i in range(nP):
                            if mask_count[i] == 0:
                                continue
                            pid = self._axis_p[i]
                            p = _products[pid]
                            for j in range(nC):
                                if self._pred_mask[i, j]:
                                    cid = self._axis_c[j]
                                    c = _customers[cid]
                                    f_csv.writerow([pid, p, cid, c, R[i, j]])
                                    pass
                                pass
                            pass
                        pass
                    f_json = open("results/results_eva_" + y + ".json", "w")
                    json.dump(self._eva, f_json)
                    f_json.close()
                    rc_result("Results exported.")
                else:
                    rc_warn("Results not exported.")
            elif opt == 6:
                # plot result
                fig = plt.figure("Predictions")
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
                            if self._pred_mask[i, j]:
                                x_pred.append(i)
                                y_pred.append(j)
                                z_pred.append(R[i, j])
                    scat_pred = ax.scatter(x_pred, y_pred, z_pred, c="m", marker=".")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.show()
            elif opt == 7:
                # cluster
                if self._X is None or self._Theta is None:
                    rc_fail("You should first optimize the model to get features!")
                else:
                    # choose an algorithm and call it
                    labels_p, labels_c = None, None
                    rc_warn("Choose an algorithm from the list below:")
                    rc_state("  1. KMeans")
                    rc_state("  2. Affinity Propagation (AP)")
                    rc_state("  3. Mean Shift")
                    y = rc_select("Choose one please: ", range_=range(1, 4), t="highlight")
                    if y == 1:
                        labels_p, labels_c = self._cluster_kmeans()
                    elif y == 2:
                        labels_p, labels_c = self._cluster_affinity()
                    elif y == 3:
                        labels_p, labels_c = self._cluster_mean_shift()
                    else:
                        rc_fail("Incorrect choice! I'll do nothing then.")
                    if labels_p is not None and labels_c is not None:
                        self._export_clusters(labels_p, labels_c, _products, _customers)
            elif opt == 8:
                # save model
                y = rc_getstr("Enter the name of this model, nothing to quit saving: ", t="highlight", keepblank=True)
                if y == "":
                    rc_warn("Quit.")
                else:
                    self.save(y)
                    rc_result("Saved Okay as '{0}'.".format(y))
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
        np.savez("data/model_" + fn, Y=self._Y, axisp=self._axis_p, axisc=self._axis_c, st=self._steps_trained, nf=self._num_features, X=self._X, Theta=self._Theta, algo=self._algo, lr=self._learning_rate, L2=self._L2)
        return

    def load(self, fn):
        rc_state("Loading model {0}".format(fn))
        npz = np.load("data/model_" + fn + ".npz")
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
