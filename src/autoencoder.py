import torch.nn as nn
import torch
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, encoder=None, decoder=None, input_shape=784, n_hidden=500,
                 z_dim=20, num_of_hidden_layers=2, dropout=0.):
        super().__init__()
        self.n_hidden = n_hidden
        self.z_dim = z_dim
        self.num_of_hidden_layers = num_of_hidden_layers
        self.input_shape = input_shape
        self.input_image_size = int(np.sqrt(input_shape))
        self.dropout = dropout

        self.flatten = nn.Flatten()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = self.encoder_builder()
            
        if decoder:
            self.decoder = decoder
        else:
            self.decoder = self.decoder_builder()
        
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)
        
    def encoder_builder(self):
        layers = []
        for i in range(self.num_of_hidden_layers + 1):
            in_features = self.input_shape if i == 0 else self.n_hidden
            out_features = self.n_hidden if i != self.num_of_hidden_layers else self.z_dim
            
            layers.append(nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(0.1),
                nn.Dropout1d(self.dropout)
            ))
        return nn.Sequential(*layers)
    
    def decoder_builder(self):
        layers = []
        for i in range(self.num_of_hidden_layers + 1):
            in_features = self.z_dim if i == 0 else self.n_hidden
            out_features = self.n_hidden if i != self.num_of_hidden_layers else self.input_shape
            layers.append(nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(0.1),
                nn.Dropout1d(self.dropout)
            ))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


class VariationalAutoEncoder(Autoencoder):
    def __init__(self, encoder=None, decoder=None, input_shape=784, n_hidden=500,
                 z_dim=20, num_of_hidden_layers=2, dropout=0.):
        super().__init__(encoder=encoder, decoder=decoder,
                         input_shape=input_shape, 
                         n_hidden=n_hidden,
                         z_dim=z_dim,
                         num_of_hidden_layers=num_of_hidden_layers,
                         dropout=dropout)
        self.Linear_mu = nn.Linear(n_hidden, z_dim)
        self.Linear_logvar = nn.Linear(n_hidden, z_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        mu = self.Linear_mu(x)
        log_var = self.Linear_logvar(x)
        
        z = self.reparameterize(mu, log_var)
        return mu, log_var, self.decoder(z)
    
    def encoder_builder(self):
        layers = []
        for i in range(self.num_of_hidden_layers):  # 마지막 output은 따로 설계해 주어야 한다. 
            in_features = self.input_shape if i == 0 else self.n_hidden
            out_features = self.n_hidden
            
            layers.append(nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout)
            ))
        return nn.Sequential(*layers)
    
    def decoder_builder(self):
        layers = []
        for i in range(self.num_of_hidden_layers + 1):
            in_features = self.z_dim if i == 0 else self.n_hidden
            out_features = self.n_hidden if i != self.num_of_hidden_layers else self.input_shape
            layers.append(nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout)
            ))
            
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def generate(self, samples):
        eps = torch.randn((samples, self.z_dim))
        return self.decoder(eps).detach().cpu().numpy()
        