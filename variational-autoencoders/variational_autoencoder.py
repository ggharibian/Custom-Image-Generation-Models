import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims: list, latent_dim: int, decoder_hidden_dims: list):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.input_size = input_dim[0] * input_dim[1] * input_dim[2]
        self.encoder_hidden_dims = encoder_hidden_dims
        self.latent_dim = latent_dim
        self.decoder_hidden_dims = decoder_hidden_dims
        
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            *self._make_encoder_layers()
        )
        
        self.decoder = nn.Sequential(
            *self._make_decoder_layers(),
            nn.Unflatten(1, self.input_dim)
        )
        
        self.mu = nn.Linear(self.encoder_hidden_dims[-1], self.latent_dim)
        self.stddev = nn.Linear(self.encoder_hidden_dims[-1], self.latent_dim)
        
    def _make_encoder_layers(self):
        layers = []
        for i, hidden_dim in enumerate(self.encoder_hidden_dims):
            if i == 0:
                layers.append(nn.Linear(self.input_size, hidden_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(self.encoder_hidden_dims[i-1], hidden_dim))
                layers.append(nn.ReLU())
        return layers
    
    def _make_decoder_layers(self):
        layers = []
        for i, hidden_dim in enumerate(self.decoder_hidden_dims):
            if i == 0:
                layers.append(nn.Linear(self.latent_dim, hidden_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(self.decoder_hidden_dims[i-1], hidden_dim))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(self.decoder_hidden_dims[-1], self.input_size))
        return layers
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        stddev = self.stddev(x)
        return mu, stddev
    
    def reparameterize(self, mu, stddev):
        eps = torch.randn_like(stddev)
        return mu + eps * stddev
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, get_loss=False):
        mu, stddev = self.encode(x)
        z = self.reparameterize(mu, stddev)
        x_hat = self.decode(z)
        if get_loss:
            loss = self.loss_function(x, x_hat, mu, stddev)
            return x_hat, loss
        else:
            return x_hat
    
    def reconstruction_loss(self, x, x_hat):
        return F.mse_loss(x_hat, x, reduction='sum')

    def kl_divergence_loss(self, mu, stddev, beta=0.5):
        return beta * torch.sum(torch.exp(stddev) + mu**2 - 1.0 - stddev)

    def loss_function(self, x, x_hat, mu, stddev, beta=0.5):
        return self.reconstruction_loss(x, x_hat) + self.kl_divergence_loss(mu, stddev, beta)
    
    # This function is not necessarily necessary, but I left it here just in case I want the decoder for something in the future.
    def get_decoder(self):
        return self.decoder