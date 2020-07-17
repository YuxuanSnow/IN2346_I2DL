"""Network base class"""
import os
import pickle
import numpy as np

"""In Pytorch you would usually define the `forward` function which performs all the interesting computations"""

from abc import ABC, abstractmethod


class Network(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define forward() method
    """
    def __init__(self, modelname='dummy_network'):
        self.modelname = modelname

    @abstractmethod
    def forward(self, X):
        """perform the forward pass through a network"""

    def __repr__(self):
        return "This is the base class for all networks we will use"

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
        ########################################################################
        # TODO                                                                 #
        # Implement the sigmoid function.                                      #
        #                                                                      #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return  1 / (1 + np.exp(-x))

    def __repr__(self):
        return "A dummy class that would compute sigmoid function"

    def save_model(self, data=None):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))
