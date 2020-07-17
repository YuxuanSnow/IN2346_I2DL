import numpy as np
"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights.

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as 
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


class SGD(object):
    def __init__(self, model, loss_func, learning_rate=1e-4):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.grads = None

    def backward(self, y_pred, y_true):
        """
        Compute the gradients wrt the weights of your model
        """
        dout = self.loss_func.backward(y_pred, y_true)
        self.model.backward(dout)

    def _update(self, w, dw, lr):
        """
        Update a model parameter
        """
        w -= lr * dw
        return w

    def step(self):
        """
        Perform an update step with the update function, using the current
        gradients of the model
        """

        # Iterate over all parameters
        for name in self.model.grads.keys():

            # Unpack parameter and gradient
            w = self.model.params[name]
            dw = self.model.grads[name]

            # Update the parameter
            w_updated = self._update(w, dw, lr=self.lr)
            self.model.params[name] = w_updated

            # Reset gradient
            self.model.grads[name] = 0.0


class sgd_momentum(object):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    def __init__(self, model, loss_func, learning_rate=1e-4, **kwargs):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.grads = None
        self.optim_config = kwargs.pop('optim_config', {})
        self._reset()

    def _reset(self):
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_configs.items()}
            self.optim_configs[p] = d

    def backward(self, y_pred, y_true):
        """
        Compute the gradients wrt the weights of your model
        """
        dout = self.loss_func.backward(y_pred, y_true)
        self.model.backward(dout)

    def _update(self, w, dw, config, lr):
        """
        Update a model parameter
        """
        if config is None:
            config = {}
        config.setdefault('momentum', 0.9)
        v = config.get('velocity', np.zeros_like(w))
        next_w = None

        ########################################################################
        # We implement the momentum update formula for you. Check it out to
        # understand it better.
        mu = config['momentum']
        learning_rate = lr
        v = mu * v - learning_rate * dw
        next_w = w + v
        config['velocity'] = v
        ########################################################################

        return next_w, config

    def step(self):
        """
        Perform an update step with the update function, using the current
        gradients of the model
        """

        # Iterate over all parameters
        for name in self.model.grads.keys():

            # Unpack parameter and gradient
            w = self.model.params[name]
            dw = self.model.grads[name]

            config = self.optim_configs[name]

            # Update the parameter
            w_updated, config = self._update(w, dw, config, lr=self.lr)
            self.model.params[name] = w_updated
            self.optim_configs[name] = config
            # Reset gradient
            self.model.grads[name] = 0.0


class Adam(object):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    def __init__(self, model, loss_func, learning_rate=1e-4, **kwargs):
        self.model = model
        self.loss_func = loss_func
        self.lr = learning_rate
        self.grads = None

        self.optim_config = kwargs.pop('optim_config', {})

        self._reset()

    def _reset(self):
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_configs.items()}
            self.optim_configs[p] = d

    def backward(self, y_pred, y_true):
        """
        Compute the gradients wrt the weights of your model
        """
        dout = self.loss_func.backward(y_pred, y_true)
        self.model.backward(dout)

    def _update(self, w, dw, config, lr):
        """
        Update a model parameter
        """
        if config is None:
            config = {}
        config.setdefault('beta1', 0.9)
        config.setdefault('beta2', 0.999)
        config.setdefault('epsilon', 1e-4)
        config.setdefault('m', np.zeros_like(w))
        config.setdefault('v', np.zeros_like(w))
        config.setdefault('t', 0)
        next_w = None

        ########################################################################
        # We  implement the momentum update formula for you. You can get a
        # better understanding how it works.
        learning_rate = lr
        m = config['m']
        v = config['v']
        t = config['t']
        beta1 = config['beta1']
        beta2 = config['beta2']
        eps = config['epsilon']

        m = beta1 * m + (1 - beta1) * dw
        m_hat = m / (1 - np.power(beta1, t + 1))
        v = beta2 * v + (1 - beta2) * (dw ** 2)
        v_hat = v / (1 - np.power(beta2, t + 1))
        next_w = w - learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        config['t'] = t + 1
        config['m'] = m
        config['v'] = v
        ########################################################################

        return next_w, config

    def step(self):
        """
        Perform an update step with the update function, using the current
        gradients of the model
        """

        # Iterate over all parameters
        for name in self.model.grads.keys():

            # Unpack parameter and gradient
            w = self.model.params[name]
            dw = self.model.grads[name]

            config = self.optim_configs[name]

            # Update the parameter
            w_updated, config = self._update(w, dw, config, lr=self.lr)
            self.model.params[name] = w_updated
            self.optim_configs[name] = config
            # Reset gradient
            self.model.grads[name] = 0.0
