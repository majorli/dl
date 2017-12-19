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

def mini_batch_gd(model, Xt, Yt, learning_rate, cost_type = 1, regu_type = 0, regu_params = None, monitor_cost = False):
    """mini-batch gradient descent
    
    one iteration of mini-batch gradient descent
    
    Arguments:
        model {Object} -- a learning model object
        Xt {array} -- one mini-batch examples
        Yt {array} -- one mini-batch labels
        learning_rate {number} -- learning rate
    
    Keyword Arguments:
        cost_type {number} -- 1 = classification, 2 = regression. Constants define in 'models' module (default: {1})
        regu_type {number} -- 0 = no regularization, 1 = L2, 2 = dropout (default: {0})
        regu_params {number of List} -- None if no regularization, lambda if L2, keep_prob list if dropout (default: {None})
        monitor_cost {bool} -- compute and return the overall cost (default: {False})
    
    Returns:
        number -- overall cost, 0.0 if monitor_cost == False
    """
    assert(model.is_ready())
    model.feed_data(Xt, Yt, init_weights = False)
    J, dW, db = model.learn(cost_type = monitor_cost * cost_type, regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
    W, b = model.get_parameters()
    for t in range(1, len(W)):
        W[t] = W[t] - learning_rate * dW[t]
        b[t] = b[t] - learning_rate * db[t]

    return J


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
