# Deep Neural Network Class
#
# Deep Neural Network is a kind of deep learning model.
# All model class should provide same interfaces and data structures.

import numpy as np

# *************** #
#    Constants    #
# *************** #

# constants for model types:
CLASSIFICATION_NN = 1
REGRESSION_NN = 2
SVM = 3
K_MEANS = 4

# constants for cost types:
ZERO = 0
LOGARITHM = 1
SQUARED_ERROR = 2

# constants for regularization types:
NONE = 0
L2 = 1
DROPOUT = 2

# *************** #
#    DL_Models    #
# *************** #

class DNN:
    """Deep neural network
    
    Deep neural nework model, for both regression nn and classification nn
    
    """

    def __init__(self):
        """constructor of object DNN (Deep Neural Network)
        
        Initialize object to invalid and not ready state, create members of object
        """
        self.n = []             # Number of units in each layer (0..L), n[0] = number of features, n[L] = dimension of the prediction
        self.g = []             # Activation function for each layer (1..L), g[0] is nothing but a placeholder, usually is None
        self.W = []             # Weights matrix of each layer (1..L), W[t].shape = (n[t], n[t-1]), W[0] = None, a placeholder
        self.b = []             # Biases matrix of each layer (1..L), b[t].shape = (n[t], 1), b[0] = None, a placeholder
        self.A = []             # List of output matrices of each layer, A[0] = X (examples), A[L] = Y_hat (predictions), A[t].shape = (n[t], m)
        self.Y = []             # Labels matrix corresponding to X, Y.shape = (n[L], m)
        self.Z = []             # Intermediate output matrix of each layer (1..L),  Z[t].shape = (n[t] * m), Z[0] = None, a placeholder

    def is_valid(self):
        """is the network valid?
        
        L = len(n) - 1 >= 2 <====> Valid
        
        Returns:
            [bool] -- network status 'is_valid'
        """
        return len(self.n) > 2

    def is_ready(self):
        """is the network ready to learn?
        
        m = A[0].shape[1] > 0 <===> Ready
        
        Returns:
            [bool] -- network status 'is_reday'
        """
        return self.is_valid() and self.A[0] is not None and self.A[0].shape[1] > 0

    def initialize(self, size, acts, weight_type = 3):
        """construct/reconstruct a deep neural network
        
        Todo: verify parameters: len(size) == len(acts) > 2
              init network structures, hyperparameters, parameters and learning data
              set state to valid, not ready
        
        Arguments:
            size {list} -- Structure of networks, length = L+1, n[t] = number of units in layer t, n[0] = n_x, n[L] = dim(Y)
            acts {list} -- Activation functions of layers, length = L+1, acts[0] = None, acts[L] = linear or sigmoid
    
        Keyword Arguments:
            weight_type {number} -- 1.0 = He init for ReLU, 2.0 = He init for Tanh, 3.0 = He init for hetero, between 0 and 1 = scale (defalut: {3})
        """
        assert(len(size) == len(acts) and len(size) > 2)
        # Network properties
        self.n = size
        self.g = acts
        self.W = [None for i in range(len(size))]
        self.b = [None for i in range(len(size))]
        # Data
        self.A = [None for i in range(len(size))]
        self.Y = []
        self.Z = [None for i in range(len(size))]
        self.initialize_parameters(weight_type)

    def initialize_parameters(self, weight_type = 3):
        """randomly initialize parameters
        
        Initialize W with normal distributed random variables, b with zeros.
        If weight scale <= 0, do He initialization

        Keyword Arguments:
            weight_type {number} -- 1.0 = He init for ReLU, 2.0 = He init for Tanh, 3.0 = He init for hetero, between 0 and 1 = scale (defalut: {3})
        """
        assert(self.is_valid())
        n = self.n
        L = len(n) - 1
        W = self.W
        b = self.b
        for t in range(1, L + 1):
            W[t] = np.random.randn(n[t], n[t - 1])
            if weight_type >= 3:
                W[t] = W[t] * ((2 / n[t - 1] + n[t]) ** 0.5)
            elif weight_type >= 2:
                W[t] = W[t] * ((1 / n[t - 1]) ** 0.5)
            elif weight_type >= 1:
                W[t] = W[t] * ((2 / n[t - 1]) ** 0.5)
            elif weight_type > 0:
                W[t] = W[t] * weight_type
            b[t] = np.zeros((n[t], 1))

    def feed_data(self, X, Y, init_weights = True, weight_type = 3):
        """feed dataset
        
        feed in dataset X and corresponding label Y.
        Todo: verify dataset: X.shape[0] == n[0]; Y.shape[0] == n[L]; X.shape[1] == Y.shape[1]
              set A[0] = X, self.Y = Y, m = X.shape[1]
              randomly initialize W if initparams == True
              set network status to 'ready'
        
        Arguments:
            X {np.ndarray} -- Matrix of examples' feature vectors, X.shape = (n[0], m)
            Y {np.ndarray} -- Matrix of labels corresponding to dataset X, Y.shape = (n[L], m)
        
        Keyword Arguments:
            init_weights {bool} -- randomly initialize weights and biases after data fed (default: {True})
            weight_type {number} -- 1.0 = He init for ReLU, 2.0 = He init for Tanh, 3.0 = He init for hetero, between 0 and 1 = scale (defalut: {3})
        """
        assert(self.is_valid())
        assert(X.shape[1] == Y.shape[1] and X.shape[0] == self.n[0] and Y.shape[0] == self.n[len(self.n) - 1])
        self.A[0] = X
        self.Y = Y
        if init_weights == True:
            self.initialize_parameters(weight_type)

    def forward_propagation(self, regu_type, keep_prob):
        """forward propagation
        
        Do forward propagation to compute predictions Y_hat = A[L]
        Do dropout regularization if retu_type == 2 and keep_prob != None

        Arguments:
            regu_type {number} -- method of regularization, 0 = no regularization, 1 = L2, 2 = dropout
            keep_prob {List} -- Dropout keep-probabilities list, L elements, do nothing if None (default: {None})

        Returns:
            List, List -- g'(z) list, Dropout masks list
        """
        n = self.n
        L = len(n) - 1
        W = self.W
        b = self.b
        g = self.g
        Z = self.Z
        A = self.A
        m = A[0].shape[1]
        D = [None for i in range(L)]        # masks for dropout
        dG = [None for i in range(L + 1)]   # g'(z) for each layer 1..L
        for t in range(1, L):
            Z[t] = np.dot(W[t], A[t - 1]) + b[t]
            A[t], dG[t] = g[t](Z[t])
            assert(A[t].shape == (n[t], m))
            if regu_type == 2 and keep_prob is not None:
                D[t] = np.random.rand(A[t].shape[0], A[t].shape[1])
                D[t] = (D[t] < keep_prob[t]) + 0
                A[t] = A[t] * D[t]
                A[t] = A[t] / keep_prob[t]
        Z[L] = np.dot(W[L], A[L - 1]) + b[L]
        A[L], dG[L] = g[L](Z[L])

        return dG, D

    def cost(self, cost_type, regu_type, lambd):
        """compute overall cost

        Compute overall cost value
        L2 regularization if regu_type == 2 and lambd > 0

        Arguments:
            cost_type {number} -- 0 = don't compute cost, 1 = logarithm cost, 2 = squared error cost
            regu_type {number} -- method of regularization, 0 = no regularization, 1 = L2, 2 = dropout
            lambd {number} -- L2 regularization parameter, less than or equals zero do nothing

        Returns:
            number -- overall cost value or 0.0 if cost_type == 0
        """
        J = 0.0
        Y = self.Y
        L = len(self.n) - 1
        Y_hat = self.A[L]
        m = self.A[0].shape[1]
        if cost_type == 1:
            # logarithm cost
            J = - np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / m
        elif cost_type == 2:
            # squared error
            J = np.sum(np.square(Y_hat - Y)) / (2 * m)

        if cost_type != 0 and regu_type == 1 and lambd > 0:
            # L2 regularization cost
            s = 0.0
            for t in range(1, L + 1):
                s = s + np.sum(np.square(self.W[t]))
            s = s * lambd / (2 * m)
            J = J + s

        return J

    def backward_propagation(self, regu_type, lambd, keep_prob, dG, D):
        """backward propagation

        Do backward propagation to compute gradients

        Arguments:
            regu_type {number} -- method of regularization, 0 = no regularization, 1 = L2, 2 = dropout
            lambd {number} -- L2 regularization parameter, less than or equals zero do nothing
            keep_prob {List} -- Dropout keep-probabilities list, L elements, do nothing if None
            dG {List} -- g'(z) list for each layer 1 to L
            D {List} -- dropout masks list for each layer 1 to L, generated by forward propagation

        Return:
            List, List -- Gradients dW, db
        """
        n = self.n
        L = len(n) - 1
        A = self.A
        Z = self.Z
        Y = self.Y
        W = self.W
        b = self.b
        m = A[0].shape[1]
        dZ = [None for i in range(L + 1)]            # Derivatives matrix (P.D. of J, w.r.t. Z) of each layer (1..L), dZ[t].shape = (n[t], m), dZ[0] = None
        dA = [None for i in range(L + 1)]            # Derivatives matrix (P.D. of J, w.r.t. A) of each layer (1..L), dA[t].shape = (n[t], m), dA[0] = None
        dW = [None for i in range(L + 1)]            # Derivatives matrix (P.D. of J, w.r.t. W) of each layer (1..L), dW[t].shape = (n[t], n[t-1]), dW[0] = None
        db = [None for i in range(L + 1)]            # Derivatives matrix (P.D. of J, w.r.t. b) of each layer (1..L), db[t].shape = (n[t], 1), db[0] = None

        dZ[L] = A[L] - Y
        dW[L] = np.dot(dZ[L], A[L - 1].T) / m
        if regu_type == 1 and lambd > 0:
            dW[L] = dW[L] + lambd / m * W[L]
        db[L] = np.mean(dZ[L], axis = 1, keepdims = True)

        for t in range(L - 1, 0, -1):
            dA[t] = np.dot(W[t + 1].T, dZ[t + 1])
            if regu_type == 2 and keep_prob is not None:
                dA[t] = dA[t] * D[t]
                dA[t] = dA[t] / keep_prob[t]
            dZ[t] = dA[t] * dG[t]
            dW[t] = np.dot(dZ[t], A[t - 1].T) / m
            if regu_type == 1 and lambd > 0:
                dW[t] = dW[t] + lambd / m * W[t]
            db[t] = np.mean(dZ[t], axis = 1, keepdims = True)

        return dW, db

    def learn(self, cost_type = 0, regu_type = 0, lambd = 0.0, keep_prob = None):
        """one iteration of learning
        
        Do one iteration of forward and backward propagation.
        
        Keyword Arguments:
            cost_type {number} -- 0 = don't compute cost, 1 = logarithm cost, 2 = squared error cost (default: {0})
            regu_type {number} -- method of regularization, 0 = no regularization, 1 = L2, 2 = dropout (default: {0})
            lambd {number} -- L2 regularization parameter, less than or equals zero do nothing (default: {0.0})
            keep_prob {List} -- Dropout keep-probabilities list, L elements, do nothing if None (default: {None})

        Returns:
            number, List, List -- Cost, derivatives of weights, derivatives of biases
        """
        assert(self.is_ready())

        dG, D = self.forward_propagation(regu_type, keep_prob)
        J = self.cost(cost_type, regu_type, lambd)
        dW, db = self.backward_propagation(regu_type, lambd, keep_prob, dG, D)

        return J, dW, db

    def predict(self, X):
        """compute predictions
        
        One pass of forward propagation without any regularization.
        Predicting will not change any data stored in the model.
        
        Arguments:
            X {np.ndarray} -- dataset
        
        Returns:
            [np.ndarray] -- predictions
        """
        assert(self.is_ready())
        assert(X.shape[0] == self.n[0])
        A = X
        W = self.W
        b = self.b
        g = self.g
        m = X.shape[1]
        for t in range(1, len(self.n)):
            Z = np.dot(W[t], A) + b[t]
            A, __temp__ = g[t](Z)
            assert(A.shape == (self.n[t], m))
        return A

    def get_parameters(self):
        """retrieve parameters
        
        return current weights and biases, client can modify them directly.
        
        Returns:
            List, List -- Weights and biases
        """
        assert(self.is_valid() and self.is_ready())
        return self.W, self.b

# ******************** #
# Activation functions #
# ******************** #


def linear(z, lbound = -np.Inf, ubound = np.Inf):
    """Bounded linear function
    
    a = z if lbound < z < ubound;
      = lbound if z <= lbound;
      = ubound if z >= ubound
    a' = 1 if lbound <= z <= ubound;
       = 0 otherwise
    
    Arguments:
        z {np.ndarray} -- input values
    
    Keyword Arguments:
        lbound {number} -- lower bound (default: {-infinite})
        ubound {number} -- upper bound (default: {infinite})

    Returns:
        np.ndarray -- a, da = a'
    """
    a = np.minimum(np.maximum(z, lbound), ubound) + 0.0
    da = ((z <= ubound) & (z >= lbound)) + 0.0
    return a, da


def identity(z):
    return linear(z)


def relu(z):
    return linear(z, lbound = 0.0)


def leaky_relu(z, slope = 0.01):
    """leaky ReLU
    
    a = max(slope * z, z)
    a' = 1 if z >= 0.0
       = slope if z < 0.0
    
    Arguments:
        z {np.ndarray} -- input values
    
    Keyword Arguments:
        slope {number} -- slope when z < 0.0, between 0,0 and 1.0 (default: {0.01})
    
    Returns:
        np.ndarray -- a, da = a'
    """
    a = np.maximum(z, slope * z) + 0.0
    da = (z >= 0) + (z < 0) * slope
    return a, da


def sigmoid(z):
    """sigmoid

    a = 1 / (1 + exp(-z))
    a' = a * (1-a)

    Arguments:
        z {np.ndarray} -- input values

    Returns:
        np.ndarray -- a, da = a'
    """
    a = 1.0 / (1 + np.exp(-z))
    da = a * (1.0 - a)

    return a, da


def tanh(z):
    """tanh

    a = tanh(z)
    a' = 1 - a ^ 2

    Arguments:
        z {np.ndarray} -- input values

    Returns:
        np.ndarray -- a, da = a'
    """
    a = np.tanh(z)
    da = 1.0 - a ** 2

    return a, da

