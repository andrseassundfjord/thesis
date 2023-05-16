
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from MVAE import MVAE
from MidMVAE import MidMVAE

class HMAE(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout = 0.1):
        super(HMAE, self).__init__()
        self.pretrained = MidMVAE(input_dims=input_dims, latent_dim=latent_dim, hidden_layers=hidden_layers, dropout=dropout)
        model_name = self.pretrained.__class__.__name__
        if input_dims[0][0] < 64:
            self.pretrained.load_state_dict(torch.load(f'augmented_models/{model_name}_state.pth'))
        else: 
            self.pretrained.load_state_dict(torch.load(f'models/{model_name}_state.pth'))

        new_input_dim = latent_dim * 2 if model_name == "MAE" else latent_dim

        self.encoder = Encoder(new_input_dim, latent_dim=latent_dim, dropout=dropout)
        self.decoder = Decoder(latent_dim=latent_dim * 2, output_dim=new_input_dim, dropout=dropout)

    def sampling(self, args):
        mu, log_var = args
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, input):
        recon_video, recon_timeseries, kl_divergence, latent_representation, mus = self.pretrained(input)
        mu, logvar = self.encoder(mus)
        z = self.sampling((mu, logvar))
        kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        z_together = torch.cat([latent_representation, z], dim = -1)
        mu_together = torch.cat([mus, mu], dim = -1)

        decoded = self.decoder(z_together)

        return mus, decoded, kl, z_together, mu_together


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout = 0.1):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(256),
            nn.Dropout(p=dropout)
        )
        self.fc31 = nn.Linear(256, latent_dim)
        self.fc32 = nn.Linear(256, latent_dim)

        for m in self.model:
            if isinstance(m, (nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = self.model(x)
        mu = self.fc31(x)
        logvar = self.fc32(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, dropout = 0.1):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(512),
            nn.Dropout(p=dropout)
        )
        self.fc3 = nn.Linear(512, output_dim)

        for m in self.model:
            if isinstance(m, (nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    
    def forward(self, x):
        x = self.model(x)
        x = self.fc3(x)
        return x