import torch.nn.functional as F
import torch.nn as nn
import pickle
import torch

class NN(nn.Module):
    '''
    Neural Network class
    Values:
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim, hidden_dim=64):
        super(NN, self).__init__()
        self.final = nn.Linear(hidden_dim * 8, 1)
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 32),
            self.make_gen_block(hidden_dim * 32, hidden_dim * 16),
            self.make_gen_block(hidden_dim * 16, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 8)
        )

    def make_gen_block(self, input_channels, output_channels):
        '''
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
        '''
        return nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, Training=True):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = features
        for layer in self.gen:
            x = layer(x)
        return torch.sigmoid(self.final(x))