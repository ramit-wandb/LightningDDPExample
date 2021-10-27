import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule

from pytorch_lightning.loggers import WandbLogger

AVAIL_GPUS = 2

# Setting the seed
pl.seed_everything(42)

# Getting the data
DATA_PATH = 'data'

train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, 
                    transform=torchvision.transforms.ToTensor())
val_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, 
                    transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

class Encoder(nn.Module):
    def __init__(self, num_input_channels : int, base_channel_size : int, latent_dim : int):
        """
        Args:
            num_input_channels (int): Number of input channels
            base_channel_size (int): Number of channels in the first convolutional layer
                                     This is also used to define the channels in the other layers.
            latent_dim (int) : Dimension of the latent vector
        """
        super(Encoder, self).__init__()

        c_hid = base_channel_size

        self.network = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, stride=2, padding=1), # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, stride=2, padding=1), # 7x7 -> 4x4
            nn.Flatten(),
            nn.Linear(4 * c_hid * 4 * 4, latent_dim),
        )
    
    def forward(self, x):
        return self.network(x)

class Decoder(nn.Module):
    def __init__(self, num_input_channels : int, base_channel_size : int, latent_dim : int):
        """
        Args:
            num_input_channels (int): Number of input channels
            base_channel_size (int): Number of channels in the first convolutional layer
                                     This is also used to define the channels in the other layers.
            latent_dim (int) : Dimensionality of latent representation
        """
        super(Decoder, self).__init__()
        c_hid = base_channel_size

        self.linear = nn.Linear(latent_dim, 4 * c_hid * 4 * 4)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                4 * c_hid, 2 * c_hid, kernel_size=2, stride=2, padding=1, output_padding=1
            ), # 4x4 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3, stride=2, padding=1, output_padding=1
            ), # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ), # 14x14 -> 28x28
            nn.Sigmoid() # Bound to [0, 1] for black to white
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        return self.network(x)

class AutoEncoder(LightningModule):
    def __init__(self, base_channel_size : int, latent_dim : int, num_input_channels : int = 1):
        '''
        Args:
            base_channel_size (int): Number of channels in the first convolutional layer
            latent_dim (int) : Dimensionality of latent representation
            num_input_channels (int): Number of input channels
        '''
        super(AutoEncoder, self).__init__()

        self.save_hyperparameters()

        self.encoder = Encoder(num_input_channels, base_channel_size, latent_dim)
        self.decoder = Decoder(num_input_channels, base_channel_size, latent_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def reconstruction_loss(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, min_lr=1e-6)

        return {
            'optimizer' : optimizer,
            'scheduler' : scheduler
        }

    def training_step(self, batch, batch_idx):
        loss = self.reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.reconstruction_loss(batch)
        self.log('val_loss', loss)
        return {'val_loss': loss}

def train(latent_dim):
    wandb_logger = WandbLogger(project='DDP-Example')

    trainer = Trainer(gpus=AVAIL_GPUS, 
                        max_epochs=10, 
                        accelerator='ddp', 
                        logger = wandb_logger)

    model = AutoEncoder(base_channel_size=28, latent_dim=latent_dim)
    trainer.fit(model, train_loader, val_loader)

    return model

def main():
    model = train(latent_dim=10)

if __name__ == '__main__':
    main()
