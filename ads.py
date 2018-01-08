# Anomaly Detection System

import numpy as np

def estimate_Gaussian(X):
    """estimate Gaussian parameters

        Estimate Gaussian parameters mu(mean), sigma2(var) for dataset X.

    Arguments:
        X -- dataset to estimate Gaussian parameters

    Returns:
        mu, sigma2 -- Gaussian parameter vectors for features in X
    """
    assert(X.shape[1] > 1 and X.shape[0] > 1)

    mu = np.mean(X, axis=1, keepdims=True)
    sigma2 = np.var(X, axis=1, keepdims=True)

    return mu, sigma2

def multi_variant_Gaussian(X, mu, sigma2):
    """multi-variant Gaussian function

        Multi-variant Gaussian density function for dataset X.

    Arguments:
        X -- dataset to compute
        mu, sigma2 -- Multi-variant Gaussian parameters, means and variances for each feature

    Returns:
        p -- results
    """
    assert(X.shape[0] == mu.shape[0])

    Xn = X - mu
    Var = np.diag(sigma2.squeeze())
    k = Xn.shape[0]
    p = ((2 * np.pi) ** (-k / 2)) * (np.linalg.det(Var) ** (-0.5)) * np.exp(-0.5 * np.sum(np.dot(np.linalg.pinv(Var), Xn) * Xn, axis=0, keepdims=True))

    return p

class ADS:
    """Anomaly Detection System"""

    def __init__(self, X):
        """Constructor

        Arguments:
            X -- dataset to detect anomalies
        """
        self.X = X
        self.mu, self.sigma2 = estimate_Gaussian(X)
        self.p = multi_variant_Gaussian(self.X, self.mu, self.sigma2)

        return

    def change_dataset(self, X):
        self.X = X
        self.mu, self.sigma2 = estimate_Gaussian(X)
        self.p = multi_variant_Gaussian(self.X, self.mu, self.sigma2)

        return

    def learn_best_threshold(self, Xval, Yval):
        """learn best threshold
            Learn best threshold for current dataset.

        Arguments:
            Xval, Yval -- Labeled validation set

        Returns:
            epsilon -- best threshold
        """
        pval = multi_variant_Gaussian(Xval, self.mu, self.sigma2)

        # TODO: 20180108 23:34
        return epsilon

    def detect(self, threshold=None, Xval=None, Yval=None):
        """detect anomalies

            Detect anomalies of current dataset

        Keyword Arguments:
            threshold -- threshold to judge anomalous, None to automatically select a best threshold by learning from Xval and Yval (default: {None})
            Xval, Yval -- labeled validate set to learn best threshold (default:{None, None})

        Returns:
            Y -- Anomalies
        """
        return Y
