import torch
import torch.nn as nn
import torch.nn.functional as F
from components import TimeseriesEncoder, TimeseriesDecoder

class TimeseriesVAE(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout):
        super(TimeseriesVAE, self).__init__()

        self.timeseries_input_shape = input_dims[1]
        self.hidden_dim = hidden_layers[1]
        self.num_hidden_layers = hidden_layers[2]
        # Encoder
        self.timeseries_encoder = TimeseriesEncoder(
            input_dim = self.timeseries_input_shape[1], 
            hidden_dim = self.hidden_dim,
            latent_dim = latent_dim,
            num_layers = self.num_hidden_layers
        )
        # Decoder
        self.timeseries_decoder = TimeseriesDecoder(
            latent_dim = latent_dim, 
            hidden_dim = self.hidden_dim, 
            output_shape = self.timeseries_input_shape,
            num_layers = self.num_hidden_layers
        )

    def forward(self, timeseries):
        # Encode timeseries
        encoded_timeseries = self.timeseries_encoder(timeseries)
        kl_divergence = -0.5 * torch.sum(1 + encoded_timeseries[2] - encoded_timeseries[1].pow(2) - encoded_timeseries[2].exp(), dim=1).mean()
        decoded_timeseries = self.timeseries_decoder(encoded_timeseries[0])
        return decoded_timeseries, kl_divergence

