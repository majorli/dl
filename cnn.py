# Comvolution Neural Network model

import numpy as np

# *************** #
# POOL Layer mode #
# *************** #

MAX_POOL = 0
AVG_POOL = 1

def num_pad(A_prev, pad, value):
    """padding channels

        Pad dimension 1 (height) and 2 (width) only.

    Arguments:
        A_prev -- Dataset, should be in shape of (m, n_H, n_W, n_C)
        pad -- padding border
        value -- padding value

    Return:
        A_prev_pad -- Padded dataset
    """
    A_prev_pad = np.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), "constant", constant_values=(value, value))
    return A_prev_pad

def single_step_conv(Slice, W, b):
    """One single step of convolution

        One single step of convolution on a slice of dataset in the same shape as the filter W

    Arguments:
        Slice -- Slice of dataset, shape (m, f, f, n_C_prev)
        W -- Filters (weights), shape (f, f, n_C_prev, n_C)
        b -- Biases, shape (1, 1, 1, n_C)

    Returns:
        Z -- Results of one single step of convolution, shape (m, n_C)
    """
    assert(Slice[0, :, :, :].shape == W[:, :, :, 0].shape)
    n_C = W.shape[3]
    assert(b.shape == (1, 1, 1, n_C))
    Z = np.zeros((Slice.shape[0], n_C))
    for c in range(n_C):
        s = Slice * W[:, :, :, c]
        Z[:, c] = np.sum(s, axis=(1, 2, 3)) + b[0, 0, 0, c]

    return Z

def conv(A_prev, W, b, pad, stride):
    """Convolution (or cross-corelation)

        Do convolution on 'm' examples with 'n_C_prev' channels by 'n_C' (f * f) filters, with zero-padding and stride

    Arguments:
        A_prev -- Dataset contains m examples, shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Filters (weights), shape (f, f, n_C_prev, n_C)
        b -- Biases, shape (1, 1, 1, n_C)
        pad -- padding border
        stride -- Stride

    Returns:
        Z -- Result of convolution, shape (m, n_H, n_W, n_C)
    """
    assert(A_prev.shape[3] == W.shape[2])
    assert(W.shape[0] == W.shape[1])
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    assert(b.shape == (1, 1, 1, n_C))

    n_H = int(np.floor((n_H_prev - f + 2 * pad) / stride) + 1)
    n_W = int(np.floor((n_W_prev - f + 2 * pad) / stride) + 1)

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = num_pad(A_prev, pad, 0.0)

    for h in range(n_H):
        for w in range(n_W):
            vert_start = h * stride
            vert_end = vert_start + f
            hori_start = w * stride
            hori_end = hori_start + f
            # apply one step of convolution
            Slice = A_prev_pad[:, vert_start:vert_end, hori_start:hori_end, :]
            Z[:, h, w, :] = single_step_conv(Slice, W, b)

    return Z

def pool(Z, f, stride, mode):
    """pooling layer

        One pass of pooling layer.

    Arguments:
        Z -- Input dataset, shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- Filter size
        stride -- Stride
        mode -- Pooling mode, MAX_POOL or AVG_POOL

    Returns:
        A -- Output of pooling, shape (m, n_H, n_W, n_C) where n_C == n_C_prev
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = Z.shape
    n_H = int(np.floor((n_H_prev - f) / stride) + 1)
    n_W = int(np.floor((n_W_prev - f) / stride) + 1)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for h in range(n_H):
        for w in range(n_W):
            vert_start = h * stride
            vert_end = vert_start + f
            hori_start = w * stride
            hori_end = hori_start + f
            # apply on step of pooling
            Slice = Z[:, vert_start:vert_end, hori_start:hori_end, :]
            if mode == AVG_POOL:
                A[:, h, w, :] = np.mean(Slice, axis=(1, 2))
            else:
                A[:, h, w, :] = np.max(Slice, axis=(1, 2))
            
    return A

def conv_backward(dZ, A_prev, W, b, pad, stride):
    """Convolution layer backward
    
        Implement the backward propagation for a convolution function
    
    Arguments:
        dZ -- gradient of the cost w.r.t the output of the conv layer (Z), shape (m, n_H, n_W, n_C)
        A_prev-- Input of the convolution layer when forward conv, shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights (Filters) of the layer, shape (f, f, n_C_prev, n_C)
        b -- Biases of the layer, shape (1, 1, 1, n_C)
        pad -- Padding border when forward conv
        stride -- Strider when forward conv
    
    Returns:
        dA_prev -- gradient of the cost w.r.t the input of the conv layer (A_prev), shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost w.r.t the weights of the conv layer (W), shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost w.r.t the biases of the conv layer (b), shape (1, 1, 1, n_C)
    """
    assert(A_prev.shape[3] == W.shape[2])
    assert(W.shape[0] == W.shape[1])
    assert(W.shape[3] == dZ.shape[3])
    assert(A_prev.shape[0] == dZ.shape[0])
    assert(b.shape[3] == W.shape[3])
    # retrieve sizes
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape

    # initialize
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # pad A_prev and dA_prev
    A_prev_pad = num_pad(A_prev, pad, 0.0)
    dA_prev_pad = num_pad(dA_prev, pad, 0.0)

    # backward convolution
    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                vert_end = vert_start + f
                hori_start = w * stride
                hori_end = hori_start + f
                a_slice = a_prev_pad[vert_start:vert_end, hori_start:hori_end, :]
                for c in range(n_C):
                    da_prev_pad[vert_start:vert_end, hori_start:hori_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = dA_prev_pad[i, pad:dA_prev_pad.shape[1]-pad, pad:dA_prev_pad.shape[2]-pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db

def _max_mask(X):
    return X == np.max(X)

def _dist_value(dz, shape):
    avg = dz / (shape[0] * shape[1])
    return np.ones(shape) * avg

def pool_backward(dA, Z, f, stride, mode):
    """
    
        Implement the backward propagation for a pooling layer
    
    Arguments:
        dA -- gradient of the cost w.r.t the output of the pooling layer, usually be computed by the backprop of next layer
        Z -- Input of the pooling layer, usually is the output of previous conv layer
        f -- pooling filter size
        stride -- Strider when forward pooling
        mode -- Pooling mode, MAX_POOL or AVG_POOL
    
    Returns:
        dZ -- gradient of cost w.r.t the input of the pooling layer
    """
    assert(Z.shape[0] == dA.shape[0])
    assert(Z.shape[3] == dA.shape[3])
    (m, n_H_prev, n_W_prev, n_C_prev) = Z.shape
    (m, n_H, n_W, n_C) = dA.shape
    
    dZ = np.zeros(Z.shape)

    for i in range(m):
        z = Z[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                vert_end = vert_start + f
                hori_start = w * stride
                hori_end = hori_start + f
                for c in range(n_C):
                    if mode == AVG_POOL:
                        dZ[i, vert_start:vert_end, hori_start:hori_end, c] += _dist_value(dA[i, h, w, c], (f, f))
                    else:
                        dZ[i, vert_start:vert_end, hori_start:hori_end, c] += _max_mask(z[vert_start:vert_end, hori_start:hori_end, c]) * dA[i, h, w, c]

    return dZ

