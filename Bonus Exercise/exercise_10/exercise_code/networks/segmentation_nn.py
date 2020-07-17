"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        num_filters = 32
        kernel_size = 3
        padding = 1
        stride = 1

        self.cnn = nn.Sequential(

            nn.Conv2d(3, 30, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(30, 60, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(60, 120, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(120, 240, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(240, 30, kernel_size=1, padding=0)
        )
        self.upsamling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (32, 120, 120)
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (32, 240, 240)
        )
        self.adjust = nn.Sequential(
            nn.Conv2d(30, 23, kernel_size=1, padding=0)
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.cnn(x)
        x = self.upsamling(x)
        x = self.adjust(x)
        #x = x.view(-1, 256 * 6 * 6)


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
