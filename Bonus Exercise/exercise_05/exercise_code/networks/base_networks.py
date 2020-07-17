"""Network base class"""
from abc import ABC, abstractmethod
import numpy as np
import os
import pickle

"""In Pytorch you would usually define the `forward` function which performs all the interesting computations"""


class Network(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define forward() method
    """

    def __init__(self, modelname="model_name"):
        self.model_name = modelname
        self.return_grad = True
        self.cache = None

    @abstractmethod
    def forward(self, X):
        """perform the forward pass through a network"""

    @abstractmethod
    def backward(self, X):
        """perform backward pass through the network (in PyTorch, this is done automatically)"""

    def __repr__(self):
        return "This is the base class for all networks we will use"

    def __call__(self, X):
        """takes data points X in train mode, and data X and output y in eval mode"""
        y = self.forward(X)
        if self.return_grad:
            return y, self.backward(y)
        else:
            return y, None

    def train(self):
        """sets the network in training mode, i.e. returns gradient when called"""
        self.return_grad = True

    def eval(self):
        """sets the network in evaluation mode, i.e. only computes forward pass"""
        self.return_grad = False

    @abstractmethod
    def save_model(self, data=None):
        """ each model should know what are the relevant things it needs for saving itself."""


class DummyNetwork(Network):
    """
    A Dummy network which takes in an input numpy array and computes its sigmoid
    """

    def __init__(self, model_name="dummy_model"):
        """
        :param modelname: A descriptive name of the model
        """
        self.model_name = model_name

    def forward(self, x):
        """
        :param x: The input to the network
        :return: results of computation of sigmoid on the input
        """
        x = 1 / (1 + np.exp(-x))

        return x

    def __repr__(self):
        return "A dummy class that would compute sigmoid function"

    def save_model(self, data=None):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' +
                                self.model_name + '.p', 'wb'))
