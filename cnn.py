# Comvolution Neural Network model

import numpy as np

# *************** #
# POOL Layer mode #
# *************** #

MAX_POOL = 0
AVG_POOL = 1

def num_pad(A_prev, p=1, v=0.0):
    """padding channels

        Pad dimension 1 (height) and 2 (width) only.

    Arguments:
        A_prev -- Dataset, should be in shape of (m, n_H, n_W, n_C)

    Keyword Arguments:
        p -- padding border (default: {1})
        v -- padding value (default: {0.0})

    Return:
        A_prev_pad -- Padded dataset
    """
    A_prev_pad = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0)), "constant", constant_values=(v, v))
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

def conv(A_prev, W, b, pad=0, stride=1):
    """Convolution (or cross-corelation)

        Do convolution on 'm' examples with 'n_C_prev' channels by 'n_C' (f * f) filters, with zero-padding and stride

    Arguments:
        A_prev -- Dataset contains m examples, shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Filters (weights), shape (f, f, n_C_prev, n_C)
        b -- Biases, shape (1, 1, 1, n_C)

    Keyword Arguments:
        pad -- padding border (default: {0})
        stride -- Stride (default: {1})

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

    A_prev_pad = num_pad(A_prev, p=pad, v=0.0)

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

def pool(Z, f, stride=1, mode=MAX_POOL):
    """pooling layer

        One pass of pooling layer.

    Arguments:
        Z -- Input dataset, shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- Filter size

    Keyword Arguments:
        stride -- Stride (default: {1})
        mode -- Pooling mode, MAX_POOL or AVG_POOL (default: {MAX_POOL})

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
    

