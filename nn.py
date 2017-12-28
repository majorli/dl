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
BOUNDED_LINEAR = 4
SOFTMAX = 6
SIGMOID = 5
LINEAR = 7

# Weight Initialization Types:
HE_RELU = 1
HE_TANH = 2
HE_OTHERS = 3

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
            act -- Activation type, RELU, TANH, LEAKY_RELU, BOUNDED_LINEAR for hidden layers and SOFTMAX, SIGMOID, LINEAR for output layers, defined by constants (default: {RELU})
            act_param -- parameter for activation function, only useful for LEAKY_RELU and BOUNDED_LINEAR (default: {None})
        """
        self.n = num_units              # number of units
        self.act = act                  # activation function type
        self.act_param = act_param      # activation function parameter(s)
        self.dropout = False            # used dropout last forward propagation
        self.A = None                   # layer final output
        return

    def forward_propagate(self, inp, keep_prob=None, batch_norm=True):
        return out

    def backward_propagate(self, inp, regu_param=None, batch_norm=True):
        return out

    def update_parameters(self, batch_norm=True):
        return

    def initialize_parameters(self, prev_layer_size, batch_norm=True, init_type=HE_RELU):
        # TODO 28/Dec/2017 14:35
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
        self.type = model_type
        self.layers = [NNLayer(dim_input, act=None)] + hidden_layers     # [Input layer, hidden layer 1, ... , hidden layer L-1, output layer L], len(self.layers) = L + 1
        if model_type == SOFTMAX_REGRESSION:
            self.layers.append(NNLayer(dim_output, SOFTMAX))
        elif model_type == LINEAR_REGRESSION:
            self.layers.append(NNLayer(dim_output, LINEAR))
        else:
            self.layers.append(NNLayer(dim_output, SIGMOID))
            self.type = LOGISTIC_REGRESSION

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

    def feed_data(self, X, normalize=False, shuffle=False):
        return
        """
