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
        self.b = None                   # vector of bias, also used as vector of beta when B.N., dim = n
        self.gamma = None               # vector of gamma when B.N., dim = 4
        self.moving_mean = 0.0          # moving mean of B.N. algorithm
        self.moving_var = 0.0           # moving variance of B.N. algorithm
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

    def forward_propagation(self, inp, keep_prob=None, batch_norm=True, bn_momentum=0.9):
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
            bn_momentum -- Momentum used to keep trace of moving means and moving variances in B.N. algorithm (default: {0.9})

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
            self.moving_mean = bn_momentum * self.moving_mean + (1 - bn_momentum) * mu
            self.moving_var = bn_momentum * self.moving_var + (1 - bn_momentum) * var
            self.t = 1.0 / np.sqrt(var + epsilon)
            self.Z_hat = (Z - mu) * self.t
            U = self.gamma * self.Z_hat + self.b
            self.A, self.dG = self.g(U)
        else:
            Z = np.dot(self.W, inp) + self.b
            self.A, self.dG = self.g(Z)

        self.dropout = (self.layer_type() == HIDDEN_LAYER) and keep_prob is not None
        if self.dropout:
            self.D = np.random.rand(self.n, m) < keep_prob
            self.A = (self.A * self.D) / keep_prob
            self.keep_prob = keep_prob

        return self.A

    def backward_propagation(self, inp, A_prev, L2_param=None, batch_norm=True):
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
            if dropout == False and L2_param is not None:
                dW = dW + L2_param / m * W
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
            if dropout == False and L2_param is not None:
                dW = dW + L2_param / m * W
            dA_prev = np.dot(W.T, dZ)

        Arguments:
            inp -- Derivatives dA, computed and sent out by the next layer
            A_prev -- Output of the previous layer, A_prev

        Keyword Arguments:
            L2_param -- L2 regularization parameter, will be ignored when dropout, None for no L2 regularization (default: {None})
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
        if self.dropout == False and L2_param is not None:
            self.dW = self.dW + L2_param / m * self.W

        dA_prev = np.dot(self.W.T, dZ)

        return dA_prev

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

    def save(self, fn):
        n = [self.layers[0].n]
        g = [None]
        W = [None]
        b = [None]
        gamma = [None]

        for l in range(1, len(self.layers)):
            n.append(self.layers[l].n)
            g.append([self.layers[l].act, self.layers[l].act_param])
            W.append(self.layers[l].W)
            b.append(self.layers[l].b)
            gamma.append(self.layers[l].gamma)

        np.savez(fn, n=n, g=g, W=W, b=b, gamma=gamma)

        return

    def load(self, fn):
        npz = np.load(fn + ".npz")
        n = list(npz["n"])
        g = list(npz["g"])
        W = list(npz["W"])
        b = list(npz["b"])
        gamma = list(npz["gamma"])

        L = len(n)
        self.layers = [NNLayer(n[0], act=None)]
        for l in range(1, L):
            layer = NNLayer(n[l], act=g[l][0], act_param=g[l][1])
            layer.W = W[l]
            layer.b = b[l]
            layer.gamma = gamma[l]
            self.layers.append(layer)

        if g[L-1][0] == SOFTMAX:
            self.model_type = SOFTMAX_REGRESSION
        elif g[L-1][0] == LINEAR:
            self.model_type = LINEAR_REGRESSION
        elif g[L-1][0] == SIGMOID:
            self.model_type = LOGISTIC_REGRESSION

        return

    def forward_propagation(self, X, keep_probs=None, batch_norm=True, bn_momentum=0.9):
        """forward propagation

            Do forward propagation to compute the final output under current parameters

        Arguments:
            X -- Input dataset

        Keyword Arguments:
            keep_probs -- List of keep probabilities for each layer from 1 to L-1, no dropout if None (default: {None})
            batch_norm -- True if use Batch Normalization (default: {True})

        Returns:
            Y_hat -- Y_hat == A[L], the final output
        """
        L = len(self.layers)
        A = X
        for l in range(1, L):
            if keep_probs is not None and l < L - 1:
                keep_prob = keep_probs[l-1]
            else:
                keep_prob = None
            A = self.layers[l].forward_propagation(A, keep_prob=keep_prob, batch_norm=batch_norm, bn_momentum=bn_momentum)
        return A

    def cost(self, Y, L2_param=None):
        """cost function

        Compute the overall cost function.

        Arguments:
            Y -- Labels

        Keyword Arguments:
            L2_param -- L2 regularization parameter, no L2 regularization if None or 0 (default: {None})

        Returns:
            J -- Overall cost
        """
        L = len(self.layers)
        A = self.layers[L-1].A
        assert(A is not None and Y is not None and A.shape == Y.shape)

        J = 0.0
        m = A.shape[1]
        if self.model_type == SOFTMAX_REGRESSION:
            J = - np.sum(Y * np.log(A)) / m
        elif self.model_type == LINEAR_REGRESSION:
            J = np.sum(np.square(A - Y)) / (2 * m)
        elif self.model_type == LOGISTIC_REGRESSION:
            J = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

        if L2_param is not None and L2_param > 0:
            s = 0.0
            for l in range(1, L):
                s = s + np.sum(np.square(self.layers[l].W))
            s = s * L2_param / (2 * m)
            J = J + s

        return J

    def backward_propagation(self, X, Y, L2_param=None, batch_norm=True):
        """backprop

        Do backward propagation to compute derivatives of cost function w.r.t. parameters in each layer.

        Arguments:
            Y -- Labels

        Keyword Arguments:
            L2_param -- L2 regularization parameter, no L2 regularization if None or 0 (default: {None})
            batch_norm -- True if use Batch Normalization (default: {True})
        """
        L = len(self.layers)
        A = self.layers[L-1].A
        assert(A is not None and Y is not None and A.shape == Y.shape)

        dA = Y
        for l in range(L-1, 1, -1):
            dA = self.layers[l].backward_propagation(dA, self.layers[l-1].A, L2_param=L2_param, batch_norm=batch_norm)
        
        self.layers[1].backward_propagation(dA, X, L2_param=L2_param, batch_norm=batch_norm)

        return

    def predict(self, X, normalize=False, mean=None, stdev=None, batch_norm=True):
        """predict

        Do prediction on given dataset X with some learned parameters. No dropout, No regularization.
        If batch normalization is not used and training set are normalized before be used to train the model, then all other dataset should be normalized by the same mean and stdev of the training set!

        Arguments:
            X -- Dataset, should be in shape (Layers[0].n, m)

        Keyword Arguments:
            normalize -- Normalize X if True (default: {False})
            mean, stdev -- Mean and standard deviation used to normalize X (default: {None, None})
            batch_norm -- True if use Batch Normalization (default: {True}) 

        Returns:
            Y -- Predictions
        """

        return Y

    def gradient_descent(self, X, Y, learning_rate, num_iters=10000, cost_step=100, keep_probs=None, L2_param=None):
        """gradient descent

        Batch gradient descent optimizer for neural network model.
        Batch Normalization is not supported by Batch Gradient Descent Optimizer.

        Arguments:
            X -- Training set
            Y -- Labels of training set
            learning_rate -- Learning rate.

        Keyword Arguments:
            num_iters -- Number of iterations (default: {10000})
            cost_step -- Step to compute cost function, 0 to never compute cost (default: {100})
            keep_probs -- List of keep probabilities for each layer from 1 to L-1, no dropout if None (default: {None})
            L2_param -- L2 regularization parameter, no L2 regularization if None or zero (default: {None})

        Returns:
            costs -- List of costs corresponding to cost_step
        """
        costs = []
        print("Start", end="", flush=True)
        print("number of iterations:" + str(num_iters))
        for i in range(num_iters):
            self.forward_propagation(X, keep_probs=keep_probs, batch_norm=False)
            if (cost_step > 0 and i % cost_step == 0) or i == num_iters - 1:
                if keep_probs is None:
                    costs.append(self.cost(Y, L2_param))
                else:
                    costs.append(self.cost(Y))
            self.backward_propagation(X, Y, L2_param=L2_param, batch_norm=False)

            for l in range(1, len(self.layers)):
                self.layers[l].W = self.layers[l].W - learning_rate * self.layers[l].dW
                self.layers[l].b = self.layers[l].b - learning_rate * self.layers[l].db

            if i % 100 == 0:
                print(".", end="", flush=True)

        print("Finished!")

        return costs

    def mini_batch_gradient_descent(self, X, Y, learning_rate, mini_batch_size=64, num_epochs=10000, cost_step=100, keep_probs=None, L2_param=None, batch_norm=True, bn_momentum=0.9):
        return costs

    def adam(self, X, Y, learning_rate, momentum=0.9, rmsprop=0.999, mini_batch_size=64, num_epochs=10000, cost_step=100, keep_probs=None, L2_param=None, batch_norm=True, bn_momentum=0.9):
        epsilon = 1e-8
        return costs
