import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator for use with a conditional GAN
class Generator(nn.Module):
    def __init__(self, layer_dims):
        super(Generator, self).__init__()
        self.layer_dims = layer_dims
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != len(layer_dims) - 2:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Tanh())
                
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.model(x)
    
# Discriminator for use with a conditional GAN
class Discriminator(nn.Module):
    def __init__(self, layer_dims):
        super(Discriminator, self).__init__()
        self.layer_dims = layer_dims
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != len(layer_dims) - 2:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Sigmoid())
                
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.model(x)
    
# class GANs(nn.Module):
#     def __init__(self, generator, discriminator) -> None:
#         super(GANs, self).__init__()
#         self.generator = generator
#         self.discriminator = discriminator
    
#     def generate(self, X):
#         return self.generator(X)
    
#     def discriminate(self, X):
#         return self.discriminator(X)
    
#     def get_generator_loss(self, X):
#         return F.binary_cross_entropy(self.discriminate(self.generate(X)), torch.zeros(X.shape[0], 1))
    
#     def get_discriminator_loss(self, X):
#         return F.binary_cross_entropy(self.discriminate(X), torch.ones(X.shape[0], 1)) + F.binary_cross_entropy(self.discriminate(self.generate(X)), torch.zeros(X.shape[0], 1))
    