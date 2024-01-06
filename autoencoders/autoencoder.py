import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dims, layer_sizes):
        super().__init__()
        self.input_dims = input_dims
        
        input_size = 1
        for x in input_dims:
            input_size *= x
        
        encoder_layers = []
        decoder_layers = []
        
        encoder_layers.append(nn.Flatten(1))
        encoder_layers.append(nn.Linear(input_size, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            decoder_layers.append(nn.Linear(layer_sizes[i+1], layer_sizes[i]))
        decoder_layers = decoder_layers[::-1]
        decoder_layers.append(nn.Linear(layer_sizes[0], input_size))
        decoder_layers.append(nn.Unflatten(1, self.input_dims))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, X, y):
        X = self.encoder(X)
        X = self.decoder(X)
        
        if y is not None:
            loss = nn.CrossEntropyLoss()
            return X, loss(X, y)
        else:
            return X, None
            