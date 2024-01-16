import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dims, layer_sizes):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dims = layer_sizes[-1]
        
        input_size = 1
        for x in input_dims:
            input_size *= x
        
        encoder_layers = []
        decoder_layers = []
        
        encoder_layers.append(nn.Flatten(1))
        encoder_layers.append(nn.Linear(input_size, layer_sizes[0]))
        encoder_layers.append(nn.ReLU())
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(layer_sizes[len(layer_sizes) - i - 1], layer_sizes[len(layer_sizes) - i - 2]))
            decoder_layers.append(nn.ReLU())
        # decoder_layers = decoder_layers[::-1]
        decoder_layers.append(nn.Linear(layer_sizes[0], input_size))
        decoder_layers.append(nn.Sigmoid())
        decoder_layers.append(nn.Unflatten(1, self.input_dims))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, X, y=None):
        X = self.encoder(X)
        X = self.decoder(X)
        
        # X is currently a probability distribution. We want to convert it to a
        # pixel value. We do this by multiplying by 255 and rounding to the
        # nearest integer.
        
        if y is not None:
            loss = nn.MSELoss()
            # print('X: ', X)
            # print('y:', y)
            # print('loss:', loss(X, y))
            # exit()
            return X, loss(X, y)
        else:
            return X, None