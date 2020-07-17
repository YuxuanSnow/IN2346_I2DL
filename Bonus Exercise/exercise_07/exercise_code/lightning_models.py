
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class TwoLayerNet(pl.LightningModule):
    def __init__(self, hparams, input_size=1 * 28 * 28, hidden_size=512, num_classes=10):
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # flatten the image first
        N, _, _, _ = x.shape
        x = x.view(N, -1)

        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        # accuracy
        _, preds = torch.max(out, 1)  # convert output probabilities to predicted class
        acc = preds.eq(targets).sum().float() / targets.size(0)

        # logs
        tensorboard_logs = {'loss': loss, 'acc': acc}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        # accuracy
        _, preds = torch.max(out, 1)
        acc = preds.eq(targets).sum().float() / targets.size(0)

        if batch_idx == 0:
            self.visualize_predictions(images, out.detach(), targets)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def visualize_predictions(self, images, preds, targets):
        class_names = ['t-shirts', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

        # determine size of the grid based on given batch size
        num_rows = torch.tensor(len(images)).float().sqrt().floor()

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(num_rows ,len(images) // num_rows + 1, i+1)
            plt.imshow(images[i].squeeze(0))
            plt.title(class_names[torch.argmax(preds, axis=-1)[i]] + f'\n[{class_names[targets[i]]}]')
            plt.axis('off')

        self.logger.experiment.add_figure('predictions', fig, global_step=self.global_step)

    def prepare_data(self):
        # download
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        fashion_mnist_train = torchvision.datasets.FashionMNIST(root='../datasets', train=True,
                                                                  download=True, transform=transform)

        fashion_mnist_test = torchvision.datasets.FashionMNIST(root='../datasets', train=False,
                                                                  download=True, transform=transform)

        # train/val split
        torch.manual_seed(0)
        train_dataset, val_dataset = random_split(fashion_mnist_train, [50000, 10000])
        torch.manual_seed(torch.initial_seed())

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = fashion_mnist_test

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(), self.hparams["learning_rate"], momentum=0.9)

        return optim
