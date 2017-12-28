# Neural network module
#
# Classes:
#   NNLayer -- neural network layer
#   NNModel -- neural network model

# ********* #
# constants #
# ********* #

# Activation Functions
RELU = 1
TANH = 2
LEAKY_RELU = 3
SIGMOID = 4
SOFTMAX = 5
IDENTITY = 6

# Layer Types
INPUT_LAYER = 0
HIDDEN_LAYER = 1
OUTPUT_LAYER = 2

# Weight Initialization Types:
HE_RELU = 1
HE_TANH = 2
HE_OTHERS = 3

# ********************** #
# Neural Network Classes #
# ********************** #

class NNLayer:
    """Neural network layer"""

    def __init__(self, num_units=2, activation=RELU, act_param=None, layer_type=HIDDEN_LAYER):
        self.n = num_units
        self.act = activation
        self.act_param = act_param
        self.type = layer_type
        return

    def forward_propagate(self, inp, dropout=None, batch_norm=True):
        return out

    def backward_propagate(self, inp, dropout=None, regu_param=None, batch_norm=True):
        return out

    def update_parameters(self, batch_norm=True):
        return

    def initialize_parameters(self, prev_layer_size, batch_norm=True, init_type=HE_RELU):
        return

class NNModel:
