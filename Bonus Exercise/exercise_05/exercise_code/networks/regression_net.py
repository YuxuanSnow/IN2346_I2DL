"""Two Layer Network."""
import numpy as np
import os
import pickle
import sys

from exercise_code.networks.layer import affine_forward, affine_backward, sigmoid_forward, sigmoid_backward
from exercise_code.networks.base_networks import Network


class RegressionNet(Network):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs regression on one value.
    """

    def __init__(self, input_size, hidden_size, std=1e-3):
        """
        :param input_size: The dimension D of the input data.
        :param hidden_size: The number of neurons H in the hidden layer.
        """
        super(RegressionNet, self).__init__("regression_net")

        self.cache = None
        np.random.seed(0)
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, 1)
        self.params['b2'] = np.zeros(1)

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.grads = {
            'W1': 0.0,
            'b1': 0.0,
            'W2': 0.0,
            'b2': 0.0
        }

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with housing prices.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        cache_affine1 = None
        cache_sigmoid = None
        cache_affine2 = None
        y = None

        ########################################################################
        # TODO                                                                 #
        # Implement the forward pass using the layers you implemented.         #
        # It consists of 3 steps:                                              #
        #   1. Forward the first affine layer                                  #
        #   2. Forward the sigmoid layer                                       #
        #   3. Forward the second affine layer                                 #
        # (Dont't forget the caches)                                           #
        ########################################################################

        out1, cache_affine1 = affine_forward(X, W1, b1)
        out2, cache_sigmoid = sigmoid_forward(out1)
        out3, cache_affine2 = affine_forward(out2, W2, b2)
        y = out3

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.cache = {
            'affine1': cache_affine1,
            'sigmoid': cache_sigmoid,
            'affine2': cache_affine2
        }

        # calculate the number of operation and memory
        batch_size = X.shape[0]
        self.num_operation = batch_size * self.input_size * self.hidden_size + \
            batch_size * self.hidden_size + batch_size * self.hidden_size * 1
        self.memory_forward = sys.getsizeof(
            cache_affine1) + sys.getsizeof(cache_affine2) + sys.getsizeof(cache_sigmoid)
        self.memory = self.memory_forward

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Unpack cache
        cache_affine1 = self.cache['affine1']
        cache_sigmoid = self.cache['sigmoid']
        cache_affine2 = self.cache['affine2']

        dW1 = None
        db1 = None
        dW2 = None
        db2 = None

        ########################################################################
        # TODO                                                                 #
        # Implement the backward pass using the layers you implemented.        #
        # Like the forward pass, it consists of 3 steps:                       #
        #   1. Backward the second affine layer                                #
        #   2. Backward the sigmoid layer                                      #
        #   3. Backward the first affine layer                                 #
        # You should now have the gradients wrt all model parameters           #
        ########################################################################

        dx2, dW2, db2 = affine_backward(dy, cache_affine2)

        dW_backup = sigmoid_backward(dx2, cache_sigmoid)
        #Why we use dx2 in sigmoid backward function?

        _, dW1, db1 = affine_backward(dW_backup, cache_affine1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.grads['W1'] = dW1
        self.grads['b1'] = db1
        self.grads['W2'] = dW2
        self.grads['b2'] = db2

        # calculate the number of operation and memory
        batch_size = dy.shape[0]
        self.num_operation = 2 * batch_size * self.input_size * self.hidden_size + \
            batch_size * self.hidden_size + 2 * batch_size * self.hidden_size * 1
        self.memory_backward = sys.getsizeof(
            dW1) + sys.getsizeof(db1) + sys.getsizeof(dW2) + sys.getsizeof(db2)
        self.memory = self.memory_forward + self.memory_backward

        return self.grads

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))
