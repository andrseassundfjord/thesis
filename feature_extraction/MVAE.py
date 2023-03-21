import torch
import torch.nn as nn
import torch.nn.functional as F
from components import TimeseriesEncoder, TimeseriesDecoder, VideoEncoder, VideoDecoder

class MVAE(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout):
        super(MVAE, self).__init__()

        self.video_input_shape = input_dims[0]
        self.timeseries_input_shape = input_dims[1]
        self.hidden_shape = hidden_layers[0]
        self.hidden_dim = hidden_layers[1]
        self.num_hidden_layers = hidden_layers[2]
        # Encoders
        self.video_encoder = VideoEncoder(latent_dim = latent_dim, 
                                            input_shape = self.video_input_shape, 
                                            hidden_shape = self.hidden_shape,
                                            dropout = dropout
                                        )
        self.timeseries_encoder = TimeseriesEncoder(
            input_dim = self.timeseries_input_shape[1], 
            hidden_dim = self.hidden_dim,
            latent_dim = latent_dim,
            num_layers = self.num_hidden_layers
        )
        # Multi-modal attention module
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * len(input_dims), latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, len(input_dims))
        )
        # Decoders
        self.video_decoder = VideoDecoder(latent_dim = latent_dim, 
                                            input_shape = self.video_input_shape, 
                                            hidden_shape = self.hidden_shape,
                                            dropout = dropout
                                        )
        self.timeseries_decoder = TimeseriesDecoder(
            latent_dim = latent_dim, 
            hidden_dim = self.hidden_dim,
            output_shape = self.timeseries_input_shape,
            num_layers = self.num_hidden_layers
        )

    def forward(self, modalities):
        # Encode each modality
        encoded_video = self.video_encoder(modalities[0])
        encoded_timeseries = self.timeseries_encoder(modalities[1])
        encoded_modalities = [encoded_video, encoded_timeseries]        
        
        # Unpack the encoded tensors and compute the Kullback-Leibler divergence loss
        zs, mus, logvars = zip(*encoded_modalities)
        kl_divergence = -0.5 * torch.sum(1 + torch.stack(logvars) - torch.stack(mus).pow(2) - torch.stack(logvars).exp(), dim=1).mean()
        
        # Compute attention weights
        flattened_modalities = torch.cat(zs, dim=1)
        attention_weights = F.softmax(self.attention(flattened_modalities), dim=1)
        
        # Compute weighted sum of modalities
        weighted_sum = torch.zeros_like(zs[0])
        for i, attention_weight in enumerate(attention_weights.split(1, dim=1)):
            weighted_sum += attention_weight * zs[i]

        # Decode latent representation
        decoded_video = self.video_decoder(weighted_sum)
        decoded_timeseries = self.timeseries_decoder(weighted_sum)
        return decoded_video, decoded_timeseries, kl_divergence

