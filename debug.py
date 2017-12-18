# Debug utilities

import numpy as np

def dnn_grad_check(model, X, Y, cost_type = 1, epsilon = 1e-7):
    """gradient check
    
    gradient checking for deep neural network models
    
    Arguments:
        model {object} -- DNN instance
        X {array} -- test cases
        Y {array} -- test cases
    
    Returns:
        number -- difference between bp gradients and numeric gradients
    """
    assert(model.is_valid())
    model.feed_data(X, Y, init_weights = False)

    J, dW, db = model.learn(cost_type = cost_type)
    g_vec = __dnn_params_list_to_vector(dW, db)

    W, b = model.get_parameters()
    p_vec = __dnn_params_list_to_vector(W, b)
    num_p = p_vec.shape[0]

    J_plus = np.zeros((num_p, 1))
    J_minus = np.zeros((num_p, 1))
    g_approx = np.zeros((num_p, 1))

    for i in range(num_p):
        p_plus = np.copy(p_vec)
        p_plus[i] = p_plus[i] + epsilon
        model.W, model.b = __dnn_params_vector_to_list(p_plus, model.n)
        _1, _2 = model.forward_propagation(0, None)
        J_plus[i] = model.cost(cost_type, 0, 0.0)
        p_minus = np.copy(p_vec)
        p_minus[i] = p_minus[i] - epsilon
        model.W, model.b = __dnn_params_vector_to_list(p_minus, model.n)
        _1, _2 = model.forward_propagation(0, None)
        J_minus[i] = model.cost(cost_type, 0, 0.0)
        g_approx[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    nume = np.linalg.norm(g_vec - g_approx)
    deno = np.linalg.norm(g_vec) + np.linalg.norm(g_approx)
    difference = nume / deno

    return difference

def __dnn_params_list_to_vector(W, b):
    n = len(W)
    vec = np.vstack((W[1].reshape(-1, 1), b[1].reshape(-1, 1)))
    for t in range(2, n):
        vec = np.vstack((vec, W[t].reshape(-1, 1), b[t].reshape(-1, 1)))
    return vec

def __dnn_params_vector_to_list(vec, n):
    W = [None]
    b = [None]
    L = len(n) - 1
    i = 0
    for t in range(1, L + 1):
        W.append(vec[i:i + n[t] * n[t-1], 0].reshape(n[t], n[t-1]))
        i = i + n[t] * n[t-1]
        b.append(vec[i: i + n[t], 0].reshape(n[t], 1))
        i = i + n[t]
    return W, b