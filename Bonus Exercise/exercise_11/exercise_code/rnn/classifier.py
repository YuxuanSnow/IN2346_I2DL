import pickle
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .rnn_nn import *
from .base_classifier import *


class RNN_Classifier(Base_Classifier):
    
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################

        self.hidden_size = hidden_size
        self.RNN = nn.RNN(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    def forward(self, x):
    ############################################################################
    #  TODO: Perform the forward pass                                          #
    ############################################################################

        batch_size = x.size()[1]
        rec, x = self.RNN(x)
        x = F.relu(self.fc1(x.reshape(batch_size, self.hidden_size)))
        x = F.relu(self.fc2(x))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return x


class LSTM_Classifier(Base_Classifier):

    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################

        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, classes)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################


    def forward(self, x):

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################

        batch_size = x.size()[1]
        rec, (x, _) = self.LSTM(x)
        x = F.relu(self.fc1(x.reshape(batch_size, self.hidden_size)))
        x = F.relu(self.fc2(x))

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return x
