# Optimization Algorithms
#  1. Gradient Descent
#  2. ...
#
#  Optimization algorithms like gradient descent take in a "ready" model instance like DNN,
#  invoke
#    `learn(cost_type = 0, regu_type = 0, lambd = 0.0, keep_prob = None)`,
#    `get_parameters()`
#  to train it.
#
#  All deep learning models should output weights, biases, gradients in a same structure
#  in a same structure: a "List" of "numpy matrices", corresponding elements should be in
#  the same shape.

def grad_desc(model, learning_rate, predict_type = 1, regu_type = 0, regu_params = None, num_iters = 5000, monitor_step = 100, print_cost = True):
    """gradient descent
    
    Optimize a model by gradient descent
    
    Arguments:
        model {Object} -- a learning model object
        learning_rate {number} -- learning rate
    
    Keyword Arguments:
        predict_type {number} -- 1 = classification, 2 = regression (default: {1})
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
            J, dW, db = model.learn(cost_type = predict_type, regu_type = regu_type, lambd = regu_params, keep_prob = regu_params)
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
