import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torchvision import transforms
from .mnist_dataset import *

class Base_Classifier(pl.LightningModule):

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass

        # out = self.forward(images)
        outputs = self.forward(images.squeeze(1).permute(1,0,2).float())
        
        # loss
        loss = F.cross_entropy(outputs, targets)

        # accuracy
        _, preds = torch.max(outputs, 1)  # convert output probabilities to predicted class
        acc = torch.mean((preds == targets).float())

        return loss, acc


    def training_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss, 'train_accuracy': acc}
        return {'loss': loss, 'train_acc':acc, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, "train")
        return {'val_loss': loss, 'val_acc': acc}
    

    def test_step(self, batch, batch_idx):
        loss, acc = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_acc': acc}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print("valiadation accuracy at currect epoch is {}".format(avg_acc))
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        with open(self.path, "rb") as f:
            mnist_raw = pickle.load(f)

        X, y= mnist_raw
        train_split=0.85

        self.train_dset=MnistDataset(X[:int(len(X)*train_split)], y[:int(len(X)*train_split)], transform=transform)
        self.val_dset=MnistDataset(X[int(len(X)*train_split):], y[int(len(X)*train_split):], transform=transform)


    @pl.data_loader
    def train_dataloader(self):
        batch_size = 128
        train_loader = torch.utils.data.DataLoader(
                        dataset=self.train_dset,
                        batch_size=batch_size,
                        shuffle=True)
         
        return train_loader


    @pl.data_loader
    def val_dataloader(self):
        batch_size = 128
        val_loader = torch.utils.data.DataLoader(
                        dataset=self.val_dset,
                        batch_size=batch_size,
                        shuffle=False)
        return val_loader

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), lr=1e-3, eps=1e-8, betas=(0.9, 0.999))
        return optim
     
    def set_data_path(self, path):
        self.path = path