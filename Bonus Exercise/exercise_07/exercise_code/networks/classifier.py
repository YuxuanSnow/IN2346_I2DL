
import os
import pickle
import numpy as np

from exercise_code.networks.base_networks import Network


class Classifier(Network):
    """
    Classifier of the form y = sigmoid(X * W)
    """
    def __init__(self, num_features=2):
        super(Classifier, self).__init__("classifier")

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
                 1-dimensional array of length N with classification scores.
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
        # Implement the forward pass and return the output of the model. Note   # 
        # that you need to implement the function self.sigmoid() for that       #  
        #########################################################################
        y = X.dot(self.W)
        y = self.sigmoid(y)


        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        return y

    def backward(self, y):
        """
        Performs the backward pass of the model.

        :param y: N x 1 array. The output of the forward pass.
        :return: Gradient of the model output (y=sigma(X*W)) wrt W
        """
        assert self.cache is not None, "run a forward pass before the backward pass"
        dW = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward pass. Return the gradient wrt W, dW              #
        # The data X is stored in self.cache. Be careful with the dimensions of   #
        # W, X and y and note that the derivative of the sigmoid fct can be       #
        # expressed by sigmoids itself (--> use the function self.sigmoid() here) # 
        ###########################################################################
        X = self.cache
        N, _ = X.shape

        # dz/dW, where z = X * W
        dW = X

        # dsigmoid/dz, where z = X * W
        dz = y * (1 - y)
        # print("dz", dz.shape)
        # print("x", X.shape)
        # dz = dz.T.repeat(N, axis=0)

        # dy/dW = dsigmoid/dz * dz/dW
        # dW = np.multiply(dW.T, np.diag(dz)).T
        dW*= dz


        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return dW

    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function

        :param x: input of the sigmoid, np.array of any shape
        :return: output of the sigmoid with same shape as input vector x
        """
        out = None
        #########################################################################
        # TODO:                                                                 #
        # Implement the sigmoid function, return out                            #
        #########################################################################
        out = 1 / (1 + np.exp(-x))
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        return out

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))
