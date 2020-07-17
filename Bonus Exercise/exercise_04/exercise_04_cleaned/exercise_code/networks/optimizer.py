# Naive Optimizer using full batch gradient descent  
import os
import pickle
import numpy as np
from exercise_code.networks.linear_model import *


class Optimizer(object):
    def __init__ (self, model, learning_rate=5e-5):
        self.model = model
        self.lr = learning_rate

    def step(self, dw):
        """
        :param dw: [D+1,1] array gradient of loss w.r.t weights of your linear model
        :return weight: [D+1,1] updated weight after one step of gradient descent  
        """
        weight = self.model.W
        #########################################################################
        # TODO:                                                                 #
        # Implement the gradient descent for 1 step to compute the weight       #
        #########################################################################
        weight = weight - self.lr * dw
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        self.model.W = weight
