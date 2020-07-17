
import os
import pickle
import numpy as np

from exercise_code.networks.base_networks import Network


class LinearModel(Network):
    """
    Linear model for regressing the housing prices.
    """
    def __init__(self, num_features=2):
        super(LinearModel, self).__init__("linear_model")

        self.num_features = num_features
        self.W = None

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        :param weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
            "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: N x D array of training data. Each row is a D-dimensional point.
        :return: Predicted labels for the data in X, shape N x 1
                 1-dimensional array of length N with housing prices.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        # save the samples for the backward pass
        self.cache = X
        # output variable
        y = None
        #########################################################################
        # TODO:                                                                 #
        # Implement the forward pass and return the output of the model.        #
        #########################################################################
        y = X @ self.W
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        return y

    def backward(self, y):
        """
        Performs the backward pass of the model.

        :param y: N x 1 array. The output of the forward pass.
        :return: Gradient of the model output (y=X*W) wrt W
        """
        assert self.cache is not None, "run a forward pass before the backward pass"
        dW = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward pass. Return the gradient wrt W, dW              #
        # The data X are stored in self.cache.                                    #
        ###########################################################################
        dW = self.cache
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        self.cache = None
        return dW

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))
