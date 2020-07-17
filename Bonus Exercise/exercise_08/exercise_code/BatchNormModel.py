import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np

class AbstractNetwork(pl.LightningModule):
    
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # load X, y to device!
        images, targets = images.to(self.device), targets.to(self.device)

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct':n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct':n_correct}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc' : acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def prepare_data(self):

        # create dataset
        fashion_mnist_train = torchvision.datasets.FashionMNIST(root='../datasets', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)
        
        fashion_mnist_test = torchvision.datasets.FashionMNIST(root='../datasets', 
                                          train=False, 
                                          transform=transforms.ToTensor())
        
        torch.manual_seed(0)
        N = len(fashion_mnist_train)
        fashion_mnist_train, fashion_mnist_val = torch.utils.data.random_split(fashion_mnist_train, 
                                                                           [int(N*0.8), int(N*0.2)])
        torch.manual_seed(torch.initial_seed())

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = fashion_mnist_train, fashion_mnist_val, fashion_mnist_test

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.batch_size)
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        return optim

    def getTestAcc(self, loader = None):
        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
        
        
class SimpleNetwork(AbstractNetwork):
    def __init__(self, hidden_dim, batch_size, learning_rate, input_size= 28 * 28, num_classes=10):
        super().__init__()

        # set hyperparams
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),            
            nn.Linear(hidden_dim, num_classes)
        )

        

    def forward(self, x):

        # x.shape = [batch_size, 1, 28, 28] -> flatten the image first
        x = x.view(x.shape[0], -1)
        
        # load to device!
        x = x.to(self.device)

        # feed x into model!
        x = self.model(x)

        return x
        

class BatchNormNetwork(AbstractNetwork):
    
    def __init__(self, hidden_dim, batch_size, learning_rate, input_size= 28 * 28, num_classes=10):
        super(BatchNormNetwork, self).__init__()
        # set hyperparams
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),            
            nn.Linear(hidden_dim, num_classes)
        )
      
    def forward(self, x):
        # x.shape = [batch_size, 1, 28, 28] -> flatten the image first
        x = x.view(x.shape[0], -1)
        
        # load to device!
        x = x.to(self.device)

        # feed x into model!
        x = self.model(x)

        return x
        

class DropoutNetwork(AbstractNetwork):
    
    def __init__(self, hidden_dim, batch_size, learning_rate, input_size= 28 * 28, num_classes=10, dropout_p=0):
        super(DropoutNetwork, self).__init__()
        
        # set hyperparams
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),   
            nn.Dropout(dropout_p),         
            nn.Linear(hidden_dim, num_classes)
        )
        
        
    def forward(self, x):
        # x.shape = [batch_size, 1, 28, 28] -> flatten the image first
        x = x.view(x.shape[0], -1)
        
        # load to device!
        x = x.to(self.device)

        # feed x into model!
        x = self.model(x)

        return x      
        
