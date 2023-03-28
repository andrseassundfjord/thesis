# Based on https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb
# From https://github.com/nicktfranklin/VAE-video/blob/master/pytorch_vae.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

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
    def __init__(self, latent_dim, input_shape, hidden_shape, dropout):
        super(VideoEncoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]
        self.hs = hidden_shape

        self.model = nn.ModuleList([
            nn.Conv3d(self.n_channel, self.hs[0] * self.n_channel, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.hs[0] * self.n_channel),
            nn.LeakyReLU(),
            nn.Conv3d(self.hs[0] * self.n_channel, self.hs[1] * self.n_channel, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.hs[1] * self.n_channel),
            nn.LeakyReLU(),
            nn.Conv3d(self.hs[1] * self.n_channel, self.hs[2] * self.n_channel, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(self.hs[2] * self.n_channel),
            nn.LeakyReLU(),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(int(self.hs[2] * self.n_channel * self.width * self.height * self.n_frames / (8 ** 3)), self.hs[3] * self.n_channel),
            nn.BatchNorm1d(self.hs[3] * self.n_channel),
            nn.LeakyReLU()
        ])
        
        self.fc_mu = nn.Linear(self.hs[3] * self.n_channel, latent_dim)
        self.fc_log_var = nn.Linear(self.hs[3] * self.n_channel, latent_dim)

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        init.kaiming_uniform_(self.fc_mu.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc_log_var.weight, mode='fan_in', nonlinearity='leaky_relu')


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        if torch.any(torch.isnan(x)):
            print("NaN in: Encoder input")
        for layer in self.model:
            x = layer(x)
            if torch.any(torch.isnan(x)):
                print("NaN in: Encoder ", layer)
            #print(layer, x.size())
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class VideoDecoder(nn.Module):
    def __init__(self, latent_dim, input_shape, hidden_shape, dropout):
        super(VideoDecoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]
        self.hs = hidden_shape

        self.model = nn.ModuleList([
            nn.Linear(latent_dim, self.hs[3] * self.n_channel),
            nn.BatchNorm1d(self.hs[3] * self.n_channel),
            nn.ReLU(),
            nn.Linear(self.hs[3] * self.n_channel, int(self.hs[2] * self.n_channel * self.width * self.height * self.n_frames / (8 ** 3))),
            nn.BatchNorm1d(int(self.hs[2] * self.n_channel * self.width * self.height * self.n_frames / (8 ** 3))),
            nn.ReLU(),
            nn.Dropout(dropout),
            Reshape((self.hs[2] * self.n_channel, int(self.n_frames / 8), int(self.width / 8), int(self.height / 8))),
            nn.ConvTranspose3d(self.hs[2] * self.n_channel, self.hs[1] * self.n_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.hs[1] * self.n_channel),
            nn.ReLU(),
            nn.ConvTranspose3d(self.hs[1] * self.n_channel, self.hs[0] * self.n_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.hs[0] * self.n_channel),
            nn.ReLU(),
            nn.ConvTranspose3d(self.hs[0] * self.n_channel, self.n_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.n_channel),
            nn.Sigmoid()
        ])

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.ConvTranspose3d, nn.Linear)):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)


    def forward(self, x):
        for layer in self.model:
            x = layer(x)
            if torch.any(torch.isnan(x)):
                print("NaN in: Decoder ", layer)
            #print(layer, x.size())
        return x

# Based on https://github.com/cerlymarco/MEDIUM_NoteBook/blob/master/VAE_TimeSeries/VAE_TimeSeries.ipynb

class TimeseriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout, num_layers=1, categorical_cols=None, embedding_dim = 16):
        super(TimeseriesEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.categorical_cols = categorical_cols
        
        if categorical_cols is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=num_cardinals, embedding_dim=embedding_dim)
                                             for num_cardinals in categorical_cols])
            self.input_dim += embedding_dim * len(self.categorical_cols)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
            
        #self.bn = nn.BatchNorm1d(hidden_dim)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x, cat_inp1 = None, cat_inp2 = None):
        """
        x: continuous features
        cat_inp1: categorical variables with 1 and 0 as possible values
        cat_inp2: categorical variables with more than two possible values
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        if self.categorical_cols is not None:
            cat_inputs = []
            for i, num_cardinals in enumerate(self.categorical_cols):
                cat_input = cat_inp2[:, :, i].long()
                cat_inputs.append(self.embeddings[i](cat_input))
            cat_inputs = torch.cat(cat_inputs, dim=-1)
            x = torch.cat([x, cat_inp1, cat_inputs], dim=-1)

        lstm_out, _ = self.lstm(x.transpose(0, 1), (h0, c0))
        #lstm_out = self.dropout(self.bn(lstm_out.transpose(1, 2)).transpose(1, 2))
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
    def __init__(self, latent_dim, hidden_dim, output_shape, dropout = 0.1, num_layers=1, categorical_cols=None, embedding_dim=16):
        super(TimeseriesDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.categorical_cols = categorical_cols
        self.embedding_dim = embedding_dim
        
        if self.categorical_cols is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(num_cardinals, self.embedding_dim) for num_cardinals in self.categorical_cols])
            self.input_dim = self.latent_dim + self.embedding_dim * len(self.categorical_cols)
        else:
            self.input_dim = self.latent_dim
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, output_shape[1])
        #self.bn = nn.BatchNorm1d(hidden_dim)
        #self.dropout = nn.Dropout(p=dropout)


    def forward(self, z, categorical_input=None):
        batch_size = z.size(0)
        seq_len = self.output_shape[0]
        
        if categorical_input is not None:
            # convert integer input to embeddings
            categorical_input = torch.tensor(categorical_input).to(z.device)
            categorical_embeddings = self.embeddings[categorical_input].transpose(1, 2)
            z = torch.cat([z.unsqueeze(1).repeat(1, seq_len, 1), categorical_embeddings], dim=-1)
        
        z = F.relu(self.fc1(z))
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(z.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(z.device)
        lstm_out, _ = self.lstm(z, (h0, c0))
        #lstm_out = self.dropout(self.bn(lstm_out))
        output = self.fc3(lstm_out)
        return output

class TimeseriesEncoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(TimeseriesEncoder2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

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

class TimeseriesDecoder2(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_shape, num_layers = 1):
        super(TimeseriesDecoder2, self).__init__()
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
