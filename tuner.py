# Functions for training and tuning models
# 
#  Optimization functions like gradient descent take in a "ready" model instance like DNN,
#  a dataset of training examples and train the model under it. 
#  
#  Evaluation functions take in a dev/test dataset and the output under this set to check
#  performance of the model under dev/test set.

import numpy as np
import math

# *************** #
#    Constants    #
# *************** #

# constants for model types:
CLASSIFICATION = 1
REGRESSION = 2

# ********************** #
# Optimization functions #
# ********************** #

def grad_desc_dnn(model, learning_rate, cost_type = 1, regu_type = 0, regu_params = None, num_iters = 10000, monitor_step = 100):
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
        monitor_step {number} -- step of monitoring cost, 0 to stop monitor costs except the first and the last iteration (default: {100})
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

    print("Start", end = "", flush = True)
    for i in range(num_iters):
        if (monitor_step > 0 and i % monitor_step == 0) or (i == 0 or i == num_iters - 1):
            J, dW, db = model.learn(cost_type = cost_type, regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
            Costs.append((i, J))
        else:
            J, dW, db = model.learn(regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
        W, b = model.get_parameters()
        for t in range(1, len(W)):
            W[t] = W[t] - learning_rate * dW[t]
            b[t] = b[t] - learning_rate * db[t]
        if (i % 100 == 0):
            print(".", end = "", flush = True)
    print("Finished!")

    return Costs

def mini_batch(model, X, Y, learning_rate, mini_batch_size = 64, cost_type = 1, regu_type = 0, regu_params = None, num_epochs = 10000, monitor_step = 100):
    """mini-batch gradient descent
    
    mini-batch gradient descent, before call this function, X and Y should be shuffled and normalized.
    
    Arguments:
        model {Object} -- a learning model object
        X {array} -- examples
        Y {array} -- labels
        learning_rate {number} -- learning rate
    
    Keyword Arguments:
        cost_type {number} -- 1 = classification, 2 = regression. Constants define in 'models' module (default: {1})
        regu_type {number} -- 0 = no regularization, 1 = L2, 2 = dropout (default: {0})
        regu_params {number or List} -- None if no regularization, lambda if L2, keep_prob list if dropout (default: {None})
        num_epochs {number} -- number of epochs (default: {10000})
        monitor_step {number} -- step of monitoring cost, 0 to stop monitor costs except the first and the last epoch (default: {100})

    Returns:
        List of tuples -- (epoch_number, costs)
    """
    assert(model.is_valid())
    Costs = []
    m = X.shape[1]
    num_batches = math.ceil(m / mini_batch_size)

    print("Start", end = "", flush = True)
    for e in range(num_epochs):
        for t in range(num_batches):
            if t == num_batches - 1:
                model.feed_data(X[:, mini_batch_size * t : ], Y[:, mini_batch_size * t : ], init_weights = False)
            else:
                model.feed_data(X[:, mini_batch_size * t : (mini_batch_size) * (t + 1)], Y[:, mini_batch_size * t : (mini_batch_size) * (t + 1)], init_weights = False)
            if t == num_batches - 1 and ((monitor_step > 0 and e % monitor_step == 0) or (e == 0 or e == num_epochs - 1)):
                J, dW, db = model.learn(cost_type = cost_type, regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
                Costs.append((e, J))
            else:
                J, dW, db = model.learn(regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
            W, b = model.get_parameters()
            for l in range(1, len(W)):
                W[l] = W[l] - learning_rate * dW[l]
                b[l] = b[l] - learning_rate * db[l]
        if (e % 100 == 0):
            print(".", end = "", flush = True)
    print("Finished!")

    return Costs


def adam(model, X, Y, learning_rate, momentum_param = 0.9, rmsprop_param = 0.999, epsilon = 1e-8, mini_batch_size = 64, cost_type = 1, regu_type = 0, regu_params = None, num_epochs = 10000, monitor_step = 100):
    """mini-batch gradient descent with Adam algorithm
    
    Adam mini-batch gradient descent, before call this function, X and Y should be shuffled and normalized.
    
    Arguments:
        model {Object} -- a learning model object
        X {array} -- examples
        Y {array} -- labels
        learning_rate {number} -- learning rate
    
    Keyword Arguments:
        momentum_param {number} -- beta1, used to compute momentum (default: {0.9})
        rmsprop_param {number} -- beta2, used to compute RMSprop (default: {0.999})
        epsilon {number} -- tiny number to avoid dividing by zero (default: {1e-8})
        cost_type {number} -- 1 = classification, 2 = regression. Constants define in 'models' module (default: {1})
        regu_type {number} -- 0 = no regularization, 1 = L2, 2 = dropout (default: {0})
        regu_params {number or List} -- None if no regularization, lambda if L2, keep_prob list if dropout (default: {None})
        num_epochs {number} -- number of epochs (default: {10000})
        monitor_step {number} -- step of monitoring cost, 0 to stop monitor costs except the first and the last epoch (default: {100})

    Returns:
        List of tuples -- (epoch_number, costs)
    """
    assert(model.is_valid())
    Costs = []
    m = X.shape[1]
    num_batches = math.ceil(m / mini_batch_size)

    VdW = [None]
    VdW_corrected = [None]
    SdW = [None]
    SdW_corrected = [None]
    Vdb = [None]
    Vdb_corrected = [None]
    Sdb = [None]
    Sdb_corrected = [None]
    W, b = model.get_parameters()
    for l in range(1, len(W)):
        VdW.append(np.zeros(W[l].shape))
        VdW_corrected.append(np.zeros(W[l].shape))
        SdW.append(np.zeros(W[l].shape))
        SdW_corrected.append(np.zeros(W[l].shape))
        Vdb.append(np.zeros(b[l].shape))
        Vdb_corrected.append(np.zeros(b[l].shape))
        Sdb.append(np.zeros(b[l].shape))
        Sdb_corrected.append(np.zeros(b[l].shape))

    count = 0
    print("Start", end = "", flush = True)
    for e in range(num_epochs):
        for t in range(num_batches):
            if t == num_batches - 1:
                model.feed_data(X[:, mini_batch_size * t : ], Y[:, mini_batch_size * t : ], init_weights = False)
            else:
                model.feed_data(X[:, mini_batch_size * t : (mini_batch_size) * (t + 1)], Y[:, mini_batch_size * t : (mini_batch_size) * (t + 1)], init_weights = False)
            if t == num_batches - 1 and ((monitor_step > 0 and e % monitor_step == 0) or (e == 0 or e == num_epochs - 1)):
                J, dW, db = model.learn(cost_type = cost_type, regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
                Costs.append((e, J))
            else:
                J, dW, db = model.learn(regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
            W, b = model.get_parameters()
            count = count + 1
            for l in range(1, len(W)):
                VdW[l] = momentum_param * VdW[l] + (1 - momentum_param) * dW[l]
                Vdb[l] = momentum_param * Vdb[l] + (1 - momentum_param) * db[l]
                VdW_corrected[l] = VdW[l] / (1 - momentum_param ** count)
                Vdb_corrected[l] = Vdb[l] / (1 - momentum_param ** count)
                
                SdW[l] = rmsprop_param * SdW[l] + (1 - rmsprop_param) * (dW[l] ** 2)
                Sdb[l] = rmsprop_param * Sdb[l] + (1 - rmsprop_param) * (db[l] ** 2)
                SdW_corrected[l] = SdW[l] / (1 - rmsprop_param ** count)
                Sdb_corrected[l] = Sdb[l] / (1 - rmsprop_param ** count)

                W[l] = W[l] - learning_rate * VdW[l] / (np.sqrt(SdW_corrected[l]) + epsilon)
                b[l] = b[l] - learning_rate * Vdb[l] / (np.sqrt(Sdb_corrected[l]) + epsilon)

        if (e % 100 == 0):
            print(".", end = "", flush = True)
    print("Finished!")

    return Costs

# ******************** #
# Evaluation functions #
# ******************** #

def evaluate(Y, Predict, model_type = 1):
    """compute precision
    
    For regressions, compute average Frobenius norm; For classification problems, compute precision.
    
    Arguments:
        Y {np.ndarray} -- labels matrix
        Predict {np.ndarray} -- predict matrix
    
    Keyword Arguments:
        model_type {number} -- 1, 2, or 3, Constants defined in 'models' module (default: {1})

    Returns:
        Dictionary -- accuracy, precision, recall, F1 for classifications:
                        {
                            "accuracy"  :   array, Y.shape[0] * 1, accuracy for each label
                            "precision" :   array, Y.shape[0] * 1, precision for each label
                            "reccall"   :   array, Y.shape[0] * 1, recall for each label
                            "F1"        :   array, Y.shape[0] * 1, F1 score for each label
                            "acc_ova"   :   number, overall accuracy
                        }
                      accuracy, Frobenius norm for regression:
                        {
                            "accuracy"  :   array, Y.shape[0] * 1, accuracy for each label
                            "worst_acc" :   array, Y.shape[0] * 1, worst accuracy for each label
                            "avg_err"   :   array, Y.shape[0] * 1, average absolute error for each label
                            "max_err"   :   array, Y.shape[0] * 1, maximum absolute error for each label
                            "Frob_norm" :   number, average Frobenius norm for all examples
                        }
    """
    assert(Y.shape == Predict.shape)
    m = Y.shape[1]
    prec = {}

    if model_type == 1:
        Yhat = (Predict >= np.amax(Predict, axis = 0, keepdims = True)) & (Predict >= 0.5)
        tp = np.sum((Y == True) & (Yhat ==True), axis = 1, keepdims = True)
        tn = np.sum((Y == False) & (Yhat == False), axis = 1, keepdims = True)
        prec["accuracy"] = (tp + tn) / m
        prec["precision"] = tp / np.sum(Yhat, axis = 1, keepdims = True)
        prec["recall"] = tp / np.sum(Y, axis = 1, keepdims = True)
        prec["F1"] = 2 * prec["precision"] * prec["recall"] / (prec["precision"] + prec["recall"])
        prec["acc_ova"] = np.sum(np.alltrue(Y == Yhat, axis = 0, keepdims = True), axis = 1, keepdims = True) / m
    elif model_type == 2:
        E = np.abs(Predict - Y)
        A = E / Y
        prec["accuracy"] = np.mean(A, axis = 1, keepdims = True)
        prec["worst_acc"] = np.amax(A, axis = 1, keepdims = True)
        prec["avg_error"] = np.mean(E, axis = 1, keepdims = True)
        prec["max_error"] = np.amax(E, axis = 1, keepdims = True)
        prec["Frob_norm"] = np.linalg.norm(E, axis = 0, keepdims = True)

    return prec
