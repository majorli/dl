# Functions for training and tuning models
# 
#  Optimization functions like gradient descent take in a "ready" model instance like DNN,
#  a dataset of training examples and train the model under it. 
#  
#  Evaluation functions take in a dev/test dataset and the output under this set to check
#  performance of the model under dev/test set.

import numpy as np

# ********************** #
# Optimization functions #
# ********************** #

def grad_desc_dnn(model, learning_rate, cost_type = 1, regu_type = 0, regu_params = None, num_iters = 5000, monitor_step = 100, print_cost = True):
    """gradient descent
    
    Optimize a model by gradient descent
    
    Arguments:
        model {Object} -- a learning model object
        learning_rate {number} -- learning rate
    
    Keyword Arguments:
        cost_type {number} -- 1 = classification, 2 = regression. Constants define in 'models' module (default: {1})
        regu_type {number} -- 0 = no regularization, 1 = L2, 2 = dropout (default: {0})
        regu_params {number of List} -- None if no regularization, lambda if L2, keep_prob list if dropout (default: {None})
        num_iters {number} -- number of iterations (default: {2500})
        monitor_step {number} -- step of monitoring cost (default: {100})
        print_cost {bool} -- print out cost values (default: {True})
    
    Returns:
        List -- List of tuples (iteration number, cost)
    """
    assert(model.is_ready())
    Costs = []
    J = 0.0
    dW = None
    db = None
    W = None
    b = None

    for i in range(num_iters):
        if (i % monitor_step == 0 or i == num_iters - 1):
            J, dW, db = model.learn(cost_type = cost_type, regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
            Costs.append((i, J))
            if (print_cost == True):
                print((i, J))
        else:
            J, dW, db = model.learn(regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
        W, b = model.get_parameters()
        for t in range(1, len(W)):
            W[t] = W[t] - learning_rate * dW[t]
            b[t] = b[t] - learning_rate * db[t]

    return Costs

# ******************** #
# Evaluation functions #
# ******************** #

def precision(Y, Predict, model_type = 1):
    """compute precision
    
    For regressions, compute average Frobenius norm; For classification problems, compute precision.
    
    Arguments:
        Y {np.ndarray} -- labels matrix
        Predict {np.ndarray} -- predict matrix
    
    Keyword Arguments:
        model_type {number} -- 1, 2, or 3, Constants defined in 'models' module (default: {1})

    Returns:
        np.ndarray -- precision
    """
    assert(Y.shape == Predict.shape)
    m = Y.shape[1]

    if model_type == 1 or model_type == 3:
        prec = np.sum((Y == Predict), axis = 1, keepdims = True) / m
    elif model_type == 2:
        prec = np.linalg.norm(Predict - Y, axis = 1, keepdims = True) / m
    else:
        prec = 0.0
    return prec
