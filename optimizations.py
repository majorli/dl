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

def grad_desc(model, learning_rate, predict_type = 1, regu_type = 0, regu_params = None, max_iters = 2500, monitor_step = 100):
	"""gradient descent
	
	Optimize a model by gradient descent
	
	Arguments:
		model {Object} -- a learning model object
		learning_rate {number} -- learning rate
	
	Keyword Arguments:
		predict_type {number} -- 1 = classification, 2 = regression (default: {1})
		regu_type {number} -- 0 = no regularization, 1 = L2, 2 = dropout (default: {0})
		regu_params {number of List} -- None if no regularization, lambda if L2, keep_prob list if dropout (default: {None})
		max_iters {number} -- max number of iterations (default: {2500})
		monitor_step {number} -- step of monitoring cost (default: {100})
	
	Returns:
		List -- List of tuples (iteration number, cost)
	"""
	return Costs
