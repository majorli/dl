# Neural network module
#
# Classes:
#   NNLayer -- neural network layer
#   NNModel -- neural network model

import numpy as np

from . import ds

# ********* #
# constants #
# ********* #

# Activation Functions
RELU = 1
TANH = 2
LEAKY_RELU = 3
SOFTMAX = 11
SIGMOID = 12
LINEAR = 13

# Weight Initialization Types:
HE_RELU = 1
HE_TANH = 2
HE_OTHERS = 3

# NN layer types
INPUT_LAYER = 1
HIDDEN_LAYER = 2
OUTPUT_LAYER = 3

# NN model types
SOFTMAX_REGRESSION = 1
LOGISTIC_REGRESSION = 2
LINEAR_REGRESSION = 3

# ********************** #
# Neural Network Classes #
# ********************** #

class NNLayer:
    """Neural network layer"""

    def __init__(self, num_units, act=RELU, act_param=None):
        """Layer instance constructor

        Layer here means hidden layer and output layer, not including input layer.

        Arguments:
            num_units -- Number of units in the layer

        Keyword Arguments:
            act -- Activation type, RELU, TANH, LEAKY_RELU for hidden layers and SOFTMAX, SIGMOID, LINEAR for output layers, defined by constants (default: {RELU})
            act_param -- parameter for activation function, only useful for LEAKY_RELU (default: {None})
        """
        self.n = num_units              # number of units
        self.act = act                  # activation function type
        self.act_param = act_param      # activation function parameter(s)
        self.dropout = False            # used dropout last forward propagation
        self.A = None                   # layer final output
        self.W = None                   # matrix of layer weights, shape = n * n_prev
        self.b = None                   # vector of bias, also used as vector of beta when BN, dim = n
        self.gamma = None               # vector of gamma when BN, dim = 4
        return

    def layer_type(self):
        """get layer type

        Layer types are defined by constants INPUT_LAYER, HIDDEN_LAYER and OUTPUT_LAYER.
        Conventionally, NNModel and its subclasses must set the activation to None for input layer (layer 0), and only choose SOFTMAX, SIGMOID or LINEAR to be the activation for output layer (layer L).

        Returns:
            t -- layer type
        """
        t = HIDDEN_LAYER
        if self.act is None:
            t = INPUT_LAYER
        elif self.act > 10:
            t = OUTPUT_LAYER

        return t

    def g(self, z):
        """activation function

        Arguments:
            z -- input

        Returns:
            a, dg -- function value and derivative at 'z'
        """
        a, dg = None, None
        if self.act == RELU:
            a, dg = ds.relu(z)
        elif self.act == TANH:
            a, dg = ds.tanh(z)
        elif self.act == LEAKY_RELU:
            a, dg = ds.leaky_relu(z, self.act_param)
        elif self.act == SIGMOID:
            a = ds.sigmoid(z)
        elif self.act == SOFTMAX:
            a = ds.softmax(z)
        elif self.act == LINEAR:
            a = ds.linear(z)
        else:
            a, dg = ds.bounded_linear(z)

        return a, dg

    def forward_propagation(self, inp, keep_prob=None, batch_norm=True):
        """forward propagation

        Forward propagation to compute the output A of layer.
        Dropout is supported if keep_prob is not None, and self.dropout will be set to True. If keep_prob is None, set self.dropout to False.
        Dropout only allowed in hidden layers. For input layer and output layer, keep_prob will be ignored and self.dropout is always set to False.
        
        Gradient descent:
            Z = np.dot(W, inp) + b;                 ## Z.shape = (n, m), m = inp.shape[1]
            A, dG = act(Z);                         ## A.shape = dG.shape = (n, m) 

        Gradient descent with Batch Normalization:
            Z = np.dot(W, inp);                     ## Z.shape = (n, m)
            mu = np.mean(Z, axis=1, keepdims=True)  ## mu.shape = (n, 1)
            var = np.var(Z, axis=1, keepdims=True)  ## var.shape = (n, 1)
            t = 1 / np.sqrt(var + epsilon)          ## t.shape = (n, 1)
            Z_hat = (Z - mu) * t                    ## Z_hat.shape = (n, m)
            U = gamma * Z_hat + b                   ## U.shape = (n, m)
            A, dG = act(U)                          ## A.shape = dG.shape = (n, m)
            
        Dropout:
            D = (np.random.rand(n, m) < keep_prob) + 0
            A = (A * D) / keep_prob

        Arguments:
            inp -- Output of the previous layer, A_prev

        Keyword Arguments:
            keep_prob -- Keep probability for dropout if hidden layer, None for no dropout (default: {None})
            batch_norm -- Use B.N. algorithm (default: {True})

        Returns:
            A -- Final output of layer
        """
        if self.layer_type() == INPUT_LAYER:
            return self.A

        assert(inp.shape[0] == self.W.shape[1])
        
        epsilon = 1e-8
        m = inp.shape[1]

        if batch_norm:
            Z = np.dot(self.W, inp)
            mu = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)
            self.t = 1.0 / np.sqrt(var + epsilon)
            self.Z_hat = (Z - mu) * self.t
            U = self.gamma * self.Z_hat + self.b
            self.A, self.dG = self.g(U)
        else:
            Z = np.dot(self.W, inp) + self.b
            self.A, self.dG = self.g(Z)

        self.dropout = (self.layer_type() == HIDDEN_LAYER) and keep_prob is not None and keep_prob > 0 and keep_prob < 1
        if self.dropout:
            self.D = np.random.rand(self.n, m) < keep_prob
            self.A = (self.A * self.D) / keep_prob
            self.keep_prob = keep_prob

        return self.A

    def backward_propagation(self, inp, A_prev, lambd=None, batch_norm=True):
        """backward propagation

        Backward propagation to compute the derivatives of cost w.r.t. the layer's parameters.
        Dropout is used last forward propagation if self.dropout is True. In this case, keep_prob is stored when forward propagation and here we will use it to rescale dA.
        
        Gradient descent:
            if output layer:                                ## inp is Y for output layer
                dZ = A - inp
            else if hidden layer:                           ## inp is dA for hidden layers
                if dropout:
                    inp = (inp * D) / keep_prob
                dZ = inp * dG
            db = np.mean(dZ, axis=1, keepdims=True)
            dW = np.dot(dZ, A_prev.T) / m
            if dropout == False and lambd is not None:
                dW = dW + lambd / m * W
            dA_prev = np.dot(W.T, dZ)

        Gradient descent with Batch Normalization:
            if output layer:                                ## inp is Y for output layer
                dU = A - inp
            else if hidden layer:                           ## inp is dA for hidden layers
                if dropout:
                    inp = (inp * D) / keep_prob
                dU = inp * dG
            dgamma = np.sum(dU * Z_hat, axis=1, keepdims=True)
            db = np.sum(dU, axis=1, keepdims=True)
            dZ = (gamma * t / m) * (m * dU - dgamma * Z_hat -  db)
            dW = np.dot(dZ, A_prev.T) / m
            if dropout == False and lambd is not None:
                dW = dW + lambd / m * W
            dA_prev = np.dot(W.T, dZ)

        Arguments:
            inp -- Derivatives dA, computed and sent out by the next layer
            A_prev -- Output of the previous layer, A_prev

        Keyword Arguments:
            lambd -- L2 regularization parameter, will be ignored when dropout, None for no L2 regularization (default: {None})
            batch_norm -- Use B.N. algorithm (default: {True})

        Returns:
            dA_prev -- derivatives of cost function w.r.t. output of previous layer
        """
        if self.layer_type() == INPUT_LAYER:
            return

        assert(A_prev.shape[0] == self.W.shape[1] and inp.shape == self.A.shape)
        m = inp.shape[1]

        if self.layer_type() == OUTPUT_LAYER:
            dU = self.A - inp
        else:
            if self.dropout:
                inp = (inp * self.D) / self.keep_prob
            dU = inp * self.dG
        if batch_norm:
            self.dgamma = np.sum(dU * self.Z_hat, axis=1, keepdims=True)
            self.db = np.sum(dU, axis=1, keepdims=True)
            dZ = (self.gamma * self.t / m) * (m * dU - self.dgamma * self.Z_hat -  self.db)
        else:
            dZ = dU
            self.db = np.mean(dZ, axis=1, keepdims=True)

        self.dW = np.dot(dZ, A_prev.T) / m
        if dropout == False and lambd is not None:
            self.dW = self.dW + lambd / m * self.W

        dA_prev = np.dot(self.W.T, dZ)

        return dA_prev

    def update_parameters(self, learning_rate, adam=True, momentum=0.9, rmsprop=0.999):
        # TODO: 20171230 23:07
        return

    def initialize_parameters(self, prev_layer_size, init_type=None, batch_norm=True):
        """initialize parameters

        Initialize parameters W, b, or W, beta, gamma by using Gaussian distributed random values

        Arguments:
            prev_layer_size -- Number of units of the previous layer

        Keyword Arguments:
            init_type -- HE_RELU, HE_TANH or HE_OTHERS, None to initialize according activation or HE_OTHERS for output layer (default {None})
            batch_norm -- Initialize BN parameters if True (default: {True})
        """
        self.b = np.zeros((self.n, 1))
    
        if batch_norm:
            self.gamma = np.ones((self.n, 1))
        else:
            self.gamma = None

        self.W = np.random.randn(self.n, prev_layer_size)
        if (init_type is None and self.act == RELU) or (init_type == HE_RELU):
            self.W = self.W * ((2 / prev_layer_size) ** 0.5)
        elif (init_type is None and self.act == TANH) or (init_type == HE_TANH):
            self.W = self.W * ((1 / prev_layer_size) ** 0.5)
        else:
            self.W = self.W * ((2 / (self.n + prev_layer_size)) ** 0.5)

        return

class NNModel:
    """Neural network model"""

    def __init__(self, dim_input, dim_output=1, hidden_layers=[], model_type=LOGISTIC_REGRESSION):
        """Neural network model instance constructor

        Create a new neural network model.

        Arguments:
            dim_input -- input data dimensions, i.e. number of features

        Keyword Arguments:
            dim_output -- output result dimensions, i.e. number of labels/classes (default: {1})
            hidden_layers -- List of layer instances for all hidden layers (default {[]})
            model_type -- Neural network type, SOFTMAX_REGRESSION, SIGMOID_REGRESSION, LINEAR_REGRESSION (default: {LOGISTIC_REGRESSION})
        """
        self.model_type = model_type        # model type
        self.layers = [NNLayer(dim_input, act=None)] + hidden_layers     # [InpLayer, HidLayer 1, ..., HidLayer L-1, OutLayer L], len(self.layers) = L+1
        if model_type == SOFTMAX_REGRESSION:
            self.layers.append(NNLayer(dim_output, SOFTMAX))
        elif model_type == LINEAR_REGRESSION:
            self.layers.append(NNLayer(dim_output, LINEAR))
        else:
            self.layers.append(NNLayer(dim_output, SIGMOID))
            self.model_type = LOGISTIC_REGRESSION

        self.Y = None                       # train/dev/test set labels

        return

    def initialize_parameters(self, init_type=None, batch_norm=True):
        """randomly initialize parameters

        Keyword Arguments:
            init_type -- HE_RELU, HE_TANH or HE_OTHERS, None to initialize according to each hidden layer and HE_OTHERS for output layer (default {None})
            batch_norm -- True to initialize parameters for batch normalization algorithm (default: {True})
        """
        for l in range(1, len(self.layers)):
            self.layers[l].initialize_parameters(self.layers[l-1].n, batch_norm=batch_norm, init_type=init_type)

        return

    def feed_data(self, X, Y, normalize=False, mean=None, stdev=None, shuffle=False):
        """feed data

        Feed data of train/dev/test set to the model, let Layers[0].A = X

        Arguments:
            X -- Examples, should be in shape (Layers[0].n, m)
            Y -- Labels, should be in shape (Layers[L].n, m)

        Keyword Arguments:
            normalize -- Normalize X if True (default: {False})
            mean, stdev -- Mean and standard deviation used to normalize X if not None (default: {None, None})
            shuffle -- Shuffle dataset if True (default: {False})

        Returns:
            mu, sigma -- Mean and standard deviation of X
        """
        assert(X.shape[0] == Y.shape[0])

        mu, sigma = None, None
        if normalize:
            self.layers[0].A, mu, sigma = ds.normalize(X, mean, stdev)
        else:
            self.layers[0].A = X

        return mu, sigma

    def predict(self, X, normalize=False, mean=None, stdev=None):
        """predict

        Do prediction on given dataset X with some learned parameters.
        No dropout, No regularization, No Batch Normalization.
        If training set are normalized before be used to train the model, then all other dataset should be normalized by the mean and stdev of the training set!

        Arguments:
            X -- Dataset, should be in shape (Layers[0].n, m)

        Keyword Arguments:
            normalize -- Normalize X if True (default: {False})
            mean, stdev -- Mean and standard deviation used to normalize X (default: {None, None})

        Returns:
            Y -- Predictions
        """

        return Y
