import numpy as np


       
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)

        x_minus_mean = x - sample_mean

        sq = x_minus_mean ** 2

        var = 1. / N * np.sum(sq, axis=0)

        sqrtvar = np.sqrt(var + eps)

        ivar = 1. / sqrtvar

        x_norm = x_minus_mean * ivar

        gammax = gamma * x_norm

        out = gammax + beta

        running_var = momentum * running_var + (1 - momentum) * var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean

        cache = (out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps)

    elif mode == 'test':
        x = (x - running_mean) / np.sqrt(running_var)
        out = x * gamma + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    out, x_norm, beta, gamma, xmu, ivar, sqrtvar, var, eps = cache

    dxnorm = dout * gamma

    divar = np.sum(dxnorm * xmu, axis=0)
    dxmu1 = dxnorm * ivar

    dsqrtvar = -1. / (sqrtvar ** 2) * divar

    dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar

    dsq = 1. / N * np.ones((N, D)) * dvar

    dxmu2 = 2 * xmu * dsq

    dx1 = dxmu1 + dxmu2
    dmean = -1. * np.sum(dx1, axis=0)

    dx2 = 1. / N * np.ones((N, D)) * dmean

    dx = dx1 + dx2

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    
    out, x_norm, beta, gamma, xmu, ivar, sqrtvar, var, eps = cache
    N, D = dout.shape

    dgamma = np.diag((dout.T).dot(x_norm))
    dbeta = np.sum(dout, axis=0)

    doutByDx = gamma * (1 - 1 / N) / ivar * (1 + 1 / N * ((out - beta) / gamma) ** 2)

    dx = dout * doutByDx

    return dx, dgamma, dbeta

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.    #
    #                                                                      #
    # HINT: You can implement spatial batch normalization using the        #
    # vanilla version of batch normalization defined above. Your           #
    # implementation should be very short; ours is less than five lines.   #
    ########################################################################

    x_s = np.transpose(x, (0, 2, 3, 1))
    #put chanel in the last. Rearrange the dims
    x_s_reshaped = np.reshape(x_s, (-1, x_s.shape[-1]))

    out_temp, cache = batchnorm_forward(x_s_reshaped, gamma, beta, bn_param)
    out = np.transpose(np.reshape(out_temp, x_s.shape), (0, 3, 1, 2))

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.   #
    #                                                                      #
    # HINT: You can implement spatial batch normalization using the        #
    # vanilla version of batch normalization defined above. Your           #
    # implementation should be very short; ours is less than five lines.   #
    ########################################################################
    dout_s = np.transpose(dout, (0, 2, 3, 1))
    dout_s_reshape = np.reshape(dout_s, (-1, dout_s.shape[-1]))

    dx_sr, dgamma, dbeta = batchnorm_backward(dout_s_reshape, cache)
    dx = np.transpose(np.reshape(dx_sr, dout_s.shape), (0, 3, 1, 2))

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return dx, dgamma, dbeta
