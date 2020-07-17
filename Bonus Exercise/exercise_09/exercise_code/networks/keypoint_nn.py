"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        num_filters = 32
        kernel_size = 3
        padding = 1
        stride = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(32, num_filters * 2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, num_filters * 4, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(128, num_filters * 8, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 30),
            nn.Tanh()
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        x = self.cnn(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    @pl.data_loader
    def train_dataloader(self, train_dataset):
        return DataLoader(train_dataset, shuffle = True, batch_size = self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self, val_dataset):
        return DataLoader(val_dataset, batch_size = self.hparams["batch_size"])

    @pl.data_loader
    def test_dataloader(self, val_dataset):
        return DataLoader(val_dataset, batch_size = self.hparams["batch_size"])

    def configure_optimizers(self):
        optim = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return optim





class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
