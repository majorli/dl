# Recommender system

import numpy as np

class Recommender:
    def __init__(self, Y, num_features=10):
        """constructor"""
        self.Y = Y
        self.normalizeRatings()
        self.initializeParameters(num_features)

        return

    def normalizeRatings(self):
        """normalize ratings
        
            Normalize the ratings, ignoring unrated elements.

        """
        assert(self.Y is not None)

        self.mean = np.nanmean(self.Y, axis=1, keepdims=True)
        self.Y_norm = self.Y - self.mean

        return

    def initializeParameters(self, num_features=10):
        """initialize parameters

            Randomly initialize parameters Theta for users, X for products.

        Keyword Arguments:
            num_features -- number of features of parameters (default: {10})
        """
        assert(self.Y is not None)

        self.X = np.random.randn(self.Y.shape[0], num_features)
        self.Theta = np.random.randn(self.Y.shape[1], num_features)

        return

    def learn(self, regu_param, cost):
        """compute cost function and gradients

            Compute the cost function and the gradients w.r.t X and Theta.

        Arguments:
            regu_param -- L2 regularization parameter
            cost -- True to compute cost function
        """
        assert(self.Y_norm is not None)
        assert(self.Theta is not None and self.X is not None)

        if regu_param is not None and regu_param > 0:
            lambd = regu_param
        else:
            lambd = 0.0

        P = np.dot(self.X, self.Theta.T)
        Diff = P - self.Y_norm
        if cost:
            J = np.nansum(Diff ** 2)
            if lambd > 0:
                J = J + lambd * (np.sum(self.X ** 2) + np.sum(self.Theta ** 2))
            self.cost = J / 2

        D = np.nan_to_num(Diff)
        self.DX = np.dot(D, self.Theta)
        self.DTheta = np.dot(D.T, self.X)
        if lambd > 0:
            self.DX = self.DX + lambd * self.X
            self.DTheta = self.DTheta + lambd * self.Theta
        
        return

    def predict(self):
        """predict on current parameters

            Predict on current parameters.

        Returns:
            P -- Predictions
            X -- Feature vetors for products
            Theta -- Feature vetors for users
        """
        P = np.dot(self.X, self.Theta.T) + self.mean

        return P, self.X.T, self.Theta.T

    def gradient_descent(self, learning_rate, num_iters=10000, cost_step=100, L2_param=None):
        """gradient descent

        Batch gradient descent optimizer for recommender model.

        Arguments:
            learning_rate -- Learning rate.

        Keyword Arguments:
            num_iters -- Number of iterations (default: {10000})
            cost_step -- Step to compute cost function, 0 to never compute cost (default: {100})
            L2_param -- L2 regularization parameter, no L2 regularization if None or zero (default: {None})

        Returns:
            costs -- List of costs corresponding to cost_step
        """
        costs = []
        print("Start", end="", flush=True)
        for i in range(num_iters):
            c = (cost_step > 0 and i % cost_step == 0) or i == num_iters - 1
            self.learn(L2_param, c)
            if c:
                costs.append((i, self.cost))

            self.X = self.X - learning_rate * self.DX
            self.Theta = self.Theta - learning_rate * self.DTheta

            if i % 100 == 0:
                print(".", end="", flush=True)

        print("Finished!")

        return costs

    def adam(self, learning_rate, momentum=0.9, rmsprop=0.999, num_iters=10000, cost_step=100, L2_param=None):
        """Adam optimizer

            Adam optimizer combines momentum and RMSProp gradient descent

        Parameters:
            learning_rate -- Learning rate.

        Keyword Parameters:
            momentum -- Momentum parameter for Adam algorithm (default: {0.9})
            rmsprop -- RMSProp parameter for Adam algorithm (default: {0.999})
            num_iters -- Number of iterations (default: {10000})
            cost_step -- Step to compute cost function, 0 to never compute cost (default: {100})
            L2_param -- L2 regularization parameter, no L2 regularization if None or zero (default: {None})

        Returns:
            costs -- List of costs corresponding to cost_step
        """
        epsilon = 1e-8
        costs = []

        VDX = np.zeros(self.X.shape)
        VDTheta = np.zeros(self.Theta.shape)
        SDX = np.zeros(self.X.shape)
        SDTheta = np.zeros(self.Theta.shape)

        print("Start", end="", flush=True)
        for i in range(num_iters):
            c = (cost_step > 0 and i % cost_step == 0) or i == num_iters - 1
            self.learn(L2_param, c)
            if c:
                costs.append((i, self.cost))

            VDX = momentum * VDX + (1 - momentum) * self.DX
            VDTheta = momentum * VDTheta + (1 - momentum) * self.DTheta
            VDX_corrected = VDX / (1 - momentum ** (i + 1))
            VDTheta_corrected = VDTheta / (1 - momentum ** (i + 1))

            SDX = rmsprop * SDX + (1 - rmsprop) * (self.DX ** 2)
            SDTheta = rmsprop * SDTheta + (1 - rmsprop) * (self.DTheta ** 2)
            SDX_corrected = SDX / (1 - rmsprop ** (i + 1))
            SDTheta_corrected = SDTheta / (1 - rmsprop ** (i + 1))

            self.X = self.X - learning_rate * VDX_corrected / (np.sqrt(SDX_corrected) + epsilon)
            self.Theta = self.Theta - learning_rate * VDTheta_corrected / (np.sqrt(SDTheta_corrected) + epsilon)

            if i % 100 == 0:
                print(".", end="", flush=True)

        print("Finished!")

        return costs

    def save(self, fn):
        np.savez(fn, Y=self.Y, X=self.X, Theta=self.Theta)
        return

    def load(self, fn):
        npz = np.load(fn + ".npz")
        self.Y = npz["Y"]
        self.X = npz["X"]
        self.Theta = npz["Theta"]
        self.normalizeRatings()
        return

    def mix_new_data(self, Y_new, avg_width=0.75):
        """mix new data

            Mix new data into current data by taking moving weighted average.

        Arguments:
            Y_new -- New data

        Keyword Arguments:
            avg_width -- Window size of moving average (default: {0.75})
        """
        assert(self.Y.shape == Y_new.shape)

        mask = np.isnan(self.Y) & ~np.isnan(Y_new)
        self.Y[mask] = Y_new[mask]
        self.Y = avg_width * self.Y + (1 - avg_width) * Y_new
        self.normalizeRatings()

        return

    def remove_users(self, indice=[]):
        self.Y = np.delete(self.Y, indice, axis=1)
        self.Theta = np.delete(self.Theta, indice, axis=0)
        self.normalizeRatings()
        return

    def remove_products(self, indice=[]):
        self.Y = np.delete(self.Y, indice, axis=0)
        self.X = np.delete(self.X, indice, axis=0)
        self.normalizeRatings()
        return

    def add_users(self, Y_new_users):
        assert(Y_new_users.shape[1] == self.Y.shape[1])
        # append data of new users and re-normalize ratings
        self.Y = np.hstack((self.Y, Y_new_users))
        self.normalizeRatings()
        # append corresponding features of new users, initialize these features randomly
        num_features = self.X.shape[1]
        self.X = np.vstack((self.X, np.random.randn(Y_new_users.shape[0], num_features)))
        
        return

    def add_products(self, Y_new_products):
        assert(Y_new_products.shape[1] == self.Y.shape[1])
        # append data of new products and re-normalize ratings
        self.Y = np.vstack((self.Y, Y_new_products))
        self.normalizeRatings()
        # append corresponding features of new products, initialize these features randomly
        num_features = self.Theta.shape[1]
        self.Theta = np.vstack((self.Theta, np.random.randn(Y_new_products.shape[0], num_features)))
        
        return
