# Neural Network Class

import numpy as np

class DNN:

    def __init__(self):
        """constructor of object DNN (Deep Neural Network)
        
        Initialize object to invalid and not ready state, create members of object
        """
        self.L = 0              # Number of layers, including the output layer but excluding the input layer
        self.n = []             # Number of units in each layer (0..L), n[0] = number of features, n[L] = dimension of the prediction
        self.g = []             # Activation function for each layer (1..L), g[0] is nothing but a placeholder, usually is None
        self.W = []             # Weights matrix of each layer (1..L), W[t].shape = (n[t], n[t-1]), W[0] = None, a placeholder
        self.b = []             # Biases matrix of each layer (1..L), b[t].shape = (n[t], 1), b[0] = None, a placeholder
        self.m = 0              # Number of examples in X
        self.A = []             # List of output matrices of each layer, A[0] = X (examples), A[L] = Y_hat (predictions), A[t].shape = (n[t], m)
        self.Y = []             # Labels matrix corresponding to X, Y.shape = (n[L], m)
        self.Z = []             # Intermediate output matrix of each layer (1..L),  Z[t].shape = (n[t] * m), Z[0] = None, a placeholder
        self.dZ = []            # Derivatives matrix (P.D. of J, w.r.t. Z) of each layer (1..L), dZ[t].shape = (n[t], m), dZ[0] = None
        self.dA = []            # Derivatives matrix (P.D. of J, w.r.t. A) of each layer (1..L), dA[t].shape = (n[t], m), dA[0] = None
        self.dW = []            # Derivatives matrix (P.D. of J, w.r.t. W) of each layer (1..L), dW[t].shape = (n[t], n[t-1]), dW[0] = None
        self.db = []            # Derivatives matrix (P.D. of J, w.r.t. b) of each layer (1..L), db[t].shape = (n[t], 1), db[0] = None
        self.J = 0.0            # Overall cost
        self.w_scale = 0.01     # Weight initial value scale, used to randomly initialize weights, default 0.01 is okay to most cases
        self.valid = False      # Network is not constructed well, can't use until call DNN.initialize() correctly
        self.ready = False      # Network has not been fed in dataset, can't learn until call DNN.feed() correctly

    def is_valid(self):
        """is the network valid?
        
        Check the network status 'is_valid'
        
        Returns:
            [bool] -- network status 'is_valid'
        """
        return self.valid

    def is_ready(self):
        """is the network ready to learn?
        
        Check the network status 'is_ready'
        
        Returns:
            [bool] -- network status 'is_reday'
        """
        return self.ready

    def initialize(self, size, acts, weight_scale = 0.01):
        """construct/reconstruct a deep neural network
        
        Todo: verify parameters: len(size) == len(acts) > 2
              init network structures, hyperparameters, parameters and learning data
              set state to valid, not ready
        
        Arguments:
            size {list} -- Structure of networks, length = L+1, n[t] = number of units in layer t, n[0] = n_x, n[L] = dim(Y)
            acts {list} -- Activation functions of layers, length = L+1, acts[0] = None, acts[L] = linear or sigmoid
    
        Keyword Arguments:
            weight_scale {number} -- weight scale, positive float, do He initialization if 0 (defalut: {0.01})
        """
        assert(len(size) == len(acts) and len(size) > 2)
        self.L = len(size) - 1
        self.n = size
        self.g = acts
        self.W = [None for i in range(self.L + 1)]
        self.b = [None for i in range(self.L + 1)]
        self.m = 0
        self.A = [None for i in range(self.L + 1)]
        self.Y = []
        self.Z = [None for i in range(self.L + 1)]
        self.dZ = [None for i in range(self.L + 1)]
        self.dA = [None for i in range(self.L + 1)]
        self.dW = [None for i in range(self.L + 1)]
        self.db = [None for i in range(self.L + 1)]
        self.J = 0.0
        self.w_scale = weight_scale
        self.initialize_parameters()
        self.valid = True
        self.ready = False

    def initialize_parameters(self):
        """randomly initialize parameters
        
        Initialize W with normal distributed random variables, b with zeros.
        If weight scale <= 0, do He initialization
        """
        assert(self.valid)
        for t in range(1, self.L + 1):
            if self.w_scale > 0:
                self.W[t] = np.random.randn(self.n[t], self.n[t - 1]) * self.w_scale
            else:
                self.W[t] = np.random.randn(self.n[t], self.n[t - 1]) * ((2 / self.n[t - 1]) ** 0.5)
            self.b[t] = np.zeros((self.n[t], 1))

    def feed_data(self, X, Y, initweights = True):
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
            initweights {bool} -- randomly initialize weights and biases after data fed (default: {True})
        """
        assert(self.valid)
        assert(X.shape[1] == Y.shape[1] and X.shape[0] == self.n[0] and Y.shape[0] == self.n[self.L])
        self.A[0] = X
        self.Y = Y
        self.m = X.shape[1]
        if initweights == True:
            self.randomly_initialize_parameters()
        self.is_ready = True

    def forward_propagation(self):
        """forward propagation
        
        Do forward propagation to compute predictions Y_hat = A[L]
        Do dropout regularization if self.regularization == 2
        """
        assert(self.valid and self.ready)
        for t in range(1, self.L):
            self.Z[t] = np.dot(self.W[t], self.A[t - 1]) + self.b[t]
            self.A[t] = self.g[t](self.Z[t])
            # assert(self.A[t].shape == (self.n[t], m))
            if self.regularization = 2:
                self.D[t] = np.random.rand(self.A[t].shape[0], self.A[t].shape[1])
                self.D[t] = (self.D[t] < self.keep_prob[t]) + 0
                self.A[t] = self.A[t] * self.D[t]
                self.A[t] = self.A[t] / self.keep_prob[t]
        self.Z[self.L] = np.dot(self.W[self.L], self.A[self.L - 1]) + self.b[self.L]
        self.A[self.L] = self.g[self.L](self.Z[self.L])

    def cost_function(self, cost):
        assert(self.valid and self.ready)
        if cost == 1:     # logarithm cost
            pass
        elif cost == 2:     # squared error
            pass

        if self.regularization == 1:
            pass

    def backward_propagation(self):
        assert(self.valid and self.ready)
        return

    def learn(self, cost = 0, regularization = 0, lambd = 0.0, keep_prob = None):
        """one iteration of learning
        
        Do one iteration of forward and backward propagation.
        
        Keyword Arguments:
            compute_cost {number} -- 0 = don't compute cost, 1 = logarithm cost, 2 = squared error cost (default: {0})
            regularization {number} -- method of regularization, 0 = no regularization, 1 = L2, 2 = dropout
            lambd {number} -- L2 regularization parameter, less than or equals zero do nothing (default: {0.0})
            keep_prob {List} -- Dropout keep-probabilities list, L elements, do nothing if None (default: {None})
        """
        assert(self.valid and self.ready)
        if regularization == 1 and lambd > 0:
            self.lambd = lambd
            self.regularization = 1
        elif regularization == 2 and keep_prob != None:
            self.keep_prob = keep_prob
            self.D = [None for i in range(self.L)]
            self.regularization = 2
        else:
            self.regularization = 0

        self.forward_propagation()
        if cost != 0:
            self.cost_function(cost)
        self.backward_propagation()

