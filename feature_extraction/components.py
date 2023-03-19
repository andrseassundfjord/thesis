# Based on https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb
# From https://github.com/nicktfranklin/VAE-video/blob/master/pytorch_vae.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

class VideoEncoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(VideoEncoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]

        self.model = nn.ModuleList([
            #nn.Conv3d(self.n_channel, 64 * self.n_channel, kernel_size = 4, stride = 2, padding=1),
            nn.Conv3d(self.n_channel, 16, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, kernel_size = 4, stride = 2, padding = 1),
            #nn.Conv3d(64 * self.n_channel, 128 * self.n_channel, kernel_size = 4, stride = 2, padding=1),
            nn.LeakyReLU(),
            #nn.Conv3d(128 * self.n_channel, 256 * self.n_channel, kernel_size = 4, stride = 2, padding=1),
            nn.Conv3d(16, 16, kernel_size = 4, stride = 2, padding = 1),
            #nn.LeakyReLU(),
            Flatten(),
            nn.Linear(256, 32),
            #nn.Linear(88 * self.n_channel * self.width * self.height, 1024 * self.n_channel),
            nn.LeakyReLU()
        ])
        
        #nn.Linear(1024 * self.n_channel, latent_dim)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
            #print(layer, x.size())
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class VideoDecoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(VideoDecoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]

        self.model = nn.ModuleList([
            #nn.Linear(latent_dim, 1024 * self.n_channel),
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            #nn.Linear(1024 * self.n_channel, 88 * self.height * self.width * self.n_channel),
            nn.ReLU(),
            Reshape((16, 4, 2, 2)),
            #Reshape((self.n_channel, self.n_frames, self.height, self.width)),
            nn.ConvTranspose3d(16, 16, kernel_size = 4, stride = 2, padding = 1),
            #nn.ConvTranspose3d(256 * self.n_channel, 128 * self.n_channel, kernel_size = 4, stride = 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 16, kernel_size = 4, stride = 2, padding = 1),
            #nn.ConvTranspose3d(128 * self.n_channel, 64 * self.n_channel, kernel_size = 4, stride = 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, self.n_channel, kernel_size = 4, stride = 2, padding = 1),
            #nn.ConvTranspose3d(64 * self.n_channel, self.n_channel, kernel_size = 4, stride = 2, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
            #print(layer, x.size())
        return x

# Based on https://github.com/cerlymarco/MEDIUM_NoteBook/blob/master/VAE_TimeSeries/VAE_TimeSeries.ipynb

class TimeseriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers = 1):
        super(TimeseriesEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.cat_emb = nn.ModuleList([])

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x.transpose(0,1), (h0, c0))
        z = F.relu(self.fc1(lstm_out[-1]))
        z_mean = self.fc_mean(z)
        z_logvar = self.fc_logvar(z)
        return self.sampling((z_mean, z_logvar)), z_mean, z_logvar

    
    def sampling(self, args):
        z_mean, z_log_sigma = args
        batch_size = z_mean.size()[0]
        epsilon = torch.randn(batch_size, self.latent_dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_sigma) * epsilon

class TimeseriesDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_shape, num_layers = 1):
        super(TimeseriesDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape
        self.num_layers = num_layers

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        #self.fc2 = nn.Linear(32, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers = num_layers, batch_first = True)
        self.fc3 = nn.Linear(hidden_dim, output_shape[1])

    def forward(self, z):
        batch_size = z.size(0)
        # Repeat for sequence length
        z = z.unsqueeze(1).repeat(1, self.output_shape[0], 1)
        z = F.relu(self.fc1(z))
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(z.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(z.device)
        lstm_out, _ = self.lstm(z, (h0, c0))
        output = self.fc3(lstm_out)
        return output
