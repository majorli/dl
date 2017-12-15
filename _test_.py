import numpy as np

def round(dims = 2, num = 100, scale = 20.0, rad = 6.0, blur = 1.0, alien = 0.02):
    X = np.random.rand(dims, num) * scale
    N = np.linalg.norm(X - scale / 2, ord = 2, axis = 0, keepdims = True)
    Y = ((N < rad - blur) | ((N >= rad - blur) & (N <= rad + blur) & (np.random.rand(N.shape[0], N.shape[1]) < 0.5))) ^ (np.random.rand(N.shape[0], N.shape[1]) < alien)
    return X, Y + 0.0

def roundn(dims = 2, num = 100, mu = 0.0, stdev = 1.0, rad = 6.0, blur = 1.0, alien = 0.02):
    X = np.random.normal(mu, stdev, (dims, num))
    N = np.linalg.norm(X - mu, ord = 2, axis = 0, keepdims = True)
    Y = ((N < rad - blur) | ((N >= rad - blur) & (N <= rad + blur) & (np.random.rand(N.shape[0], N.shape[1]) < 0.5))) ^ (np.random.rand(N.shape[0], N.shape[1]) < alien)
    return X, Y + 0.0
