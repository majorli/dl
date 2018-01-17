import numpy as np
import tensorflow as tf

def __normalizeRatings(Y):
    """
    
    Normalize the ratings, ignoring unrated elements.
    
    Arguments:
        Y -- np.array, shape (num_products, num_customers)

    Returns:
        Y_mean -- np.array, average ratings by products, shape (num_products, 1)
        Y_norm -- np.array, normalized ratings, shape(num_products, num_customers)
    """
    Y_mean = np.nanmean(Y, axis=1, keepdims=True)
    Y_norm = Y - Y_mean

    return Y_mean, Y_norm

def __recommend(X, Theta, Y_mean):
    """

    Compute recommendation

    Arguments:
        X -- features of products, shape (num_products, num_features)
        Theta -- features of customers, shape (num_customers, num_features)
        Y_mean -- average ratings by products, shape (num_products, 1)

    Returns:
        R -- recommendation, np.array, shape (num_products, num_customers)
    """
    R = np.round(np.dot(X, Theta.T) + Y_mean)

    return R

def learn(Y, num_features=100, adam=False, learning_rate=0.0001, lambd=0.0, num_steps=1000, cost_step=10):
    """

    Run the model to fit recommendation

    Arguments:
        Y -- dataset to learn, np.array, shape (num_products, num_customers)
    
    Keyword Arguments:
        num_features -- number of features to learn (default: {100})
        adam -- True to use adam optimizer (default: {False})
        learning_rate -- learning rate (default: {0.0001})
        lamdb -- L2 regularization parameter (default: {0.0})
        num_steps -- number of learning steps (default: {1000})
        cost_step -- step to print cost (default: {10})

    Returns:
        R -- recommendation, np.array, shape (num_products, num_customers)
        X -- product features, np.array, shape (num_products, num_features)
        Theta -- customer features, np.array, shape (num_customers, num_features)
        Y_mean -- average ratings by products, shape (num_products, 1)
    """
    (num_products, num_customers) = Y.shape
    Y_mean, Y_norm = __normalizeRatings(Y)

    # constant: L2 regularization parameter
    l2 = tf.constant(lambd)

    # placeholder: dataset to learn
    y = tf.placeholder(tf.float32, shape=[num_products, num_customers])

    # Variables: feature vectors for all customers (theta) and all products (x)
    x = tf.Variable(tf.random_normal([Y.shape[0], num_features]))
    theta = tf.Variable(tf.random_normal([Y.shape[1], num_features]))

    # computation graph
    pred = tf.matmul(x, tf.transpose(theta))
    diff = pred - y
    d = tf.where(tf.is_nan(diff), tf.zeros_like(diff), diff)

    # cost function
    cost = (tf.reduce_sum(tf.square(d)) + l2 * (tf.reduce_sum(tf.square(x)) + tf.reduce_sum(tf.square(theta)))) / 2.0

    # optimizer
    if adam:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # variable initialzer
    init = tf.global_variables_initializer()

    # learn
    sess = tf.Session()
    sess.run(init)

    for t in range(num_steps):
        _, c = sess.run([optimizer, cost], feed_dict ={y:Y_norm})
        if t % cost_step == 0:
            print("step: ", t, ", cost: ", c)
    
    # recommend
    X = sess.run(x)
    Theta = sess.run(theta)
    R = __recommend(X, Theta, Y_mean)

    sess.close()

    return R, X, Theta, Y_mean

