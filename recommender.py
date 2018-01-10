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
        assert(self.Y_norm is not)
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
            self.J = J / 2

        D = np.nan_to_num(Diff)
        self.DX = np.dot(D, self.Theta)
        self.DTheta = np.dot(D.T, self.X)
        if lamdb > 0:
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
