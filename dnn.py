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
        self.reg_param = 0.0    # Regularization parameter, defalut 0.0 (no regularization)
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

    def initialize(self, size, acts, reg_param = 0.0, weight_scale = 0.01):
        """construct/reconstruct a deep neural network
        
        Todo: verify parameters: len(size) == len(acts) > 2
              init network structures, hyperparameters, parameters and learning data
              set state to valid, not ready
        
        Arguments:
            size {list} -- Structure of networks, length = L+1, n[t] = number of units in layer t, n[0] = n_x, n[L] = dim(Y)
            acts {list} -- Activation functions of layers, length = L+1, acts[0] = None, acts[L] = linear or sigmoid
    
        Keyword Arguments:
            reg_param {number} -- regularization parameter 'lambda', positive float (default: {0.0})
            weight_scale {number} -- weight scale, positive float, (defalut: {0.01})
        """
        assert(len(size) == len(acts) and len(size) > 2)
        self.L = len(size) - 1
        self.n = size
        self.g = acts
        self.W = [None]
        self.b = [None]
        self.m = 0
        self.A = [None]
        self.Y = []
        self.Z = [None]
        self.dZ = [None]
        self.dA = [None]
        self.dW = [None]
        self.db = [None]
        self.J = 0.0
        for t in range(1, self.L + 1):
            self.W.append(np.random.randn(self.n[t], self.n[t - 1]) * self.w_scale)
            self.b.append(np.zeros((self.n[t], 1)))
            self.A.append(None)
            self.Z.append(None)
            self.dZ.append(None)
            self.dA.append(None)
            self.dW.append(None)
            self.db.append(None)
        self.reg_param = reg_param
        self.w_scale = weight_scale
        self.valid = True
        self.ready = False

    def initialize_parameters(self):
        """randomly initialize parameters
        
        Initialize W with normal distributed random variables, b with zeros.
        """
        assert(self.valid)
        for t in range(1, self.L + 1):
            self.W[t] = np.random.randn(self.n[t], self.n[t - 1]) * self.w_scale
            self.b[t] = np.zeros((self.n[t], 1))


    def set_regularization_parameter(self, reg_param):
        """set regularization parameter
        
        set new regularization parameter, then initialize parameters (weights and biases)
        
        Arguments:
            reg_param {[type]} -- [description]
        """
        assert(self.valid)
        self.reg_param = reg_param
        self.randomly_initialize_parameters()

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
        assert(self.valid and self.ready)
        return

    def cost_function(self):
        assert(self.valid and self.ready)
        return

    def backward_propagation(self):
        assert(self.valid and self.ready)
        return

    def learn(self, computez_cost = True):
        assert(self.valid and self.ready)
        return
