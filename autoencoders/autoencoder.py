import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        encoder_layers = []
        decoder_layers = []
        
        encoder_layers.append(nn.Linear(input_size, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            decoder_layers.append(nn.Linear(layer_sizes[i+1], layer_sizes[i]))
        decoder_layers.append(nn.Linear(layer_sizes[-1], input_size))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
            