import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k).

    We will reshape each input into a vector of dimension D = d_1 * ... * d_k,
    and then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    out = None
    ########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.    #
    # You will need to reshape the input into rows.                        #
    ########################################################################

    N = x.shape[0]
    D_backup = x.shape[1:]

    D = 1
    for i in range(len(D_backup)):
        D = D * D_backup[i]

    x_new = np.reshape(x, (N, D))

    out = np.dot(x_new, w) + b
    # x_new@w shape is N*M, N is number of samples, M is the number of outputs and biases.
    # b is M*1, because for each N, the bias is the same. We don't need to let the dimension be N*M
    # a = [[2, 2, 2], [3, 3, 3]]
    # b = [10, 10, 10]
    # np.asarray(a) + np.asarray(b)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,)

    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the affine backward pass.                            #
    # Hint: Don't forget to average the gradients dw and db                #
    ########################################################################
    N = x.shape[0]
    D_backup = x.shape[1:]

    D = 1
    for i in range(len(D_backup)):
        D = D * D_backup[i]

    x_new = np.reshape(x, (N, D))

    dw = np.dot(x_new.T, dout) / len(x)
    db = np.sum(dout, axis=0) / len(x)

    dx_backup = np.dot(dout, w.T)
    dx = np.reshape(dx_backup, x.shape)
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx, dw, db


def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of sigmoids.

    :param x: Inputs, of any shape

    :return out: Output, of the same shape as x
    :return cache: out
    """
    out = None
    ########################################################################
    # TODO: Implement the Sigmoid forward pass.                            #
    ########################################################################

    out = 1 / (1 + np.exp(-x))

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = out
    return out, cache


def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for a layer of sigmoids.

    :param dout: Upstream derivatives, of any shape
    :param cache: Output y of the forward pass, of same shape as dout

    :return dx: Gradient with respect to x
    """
    dx = None
    y = cache
    ########################################################################
    # TODO: Implement the Sigmoid backward pass.                           #
    ########################################################################

    sigmoid_derivative = y * (1-y)
    dx = dout * sigmoid_derivative

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx
