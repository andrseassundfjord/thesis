import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class MVAE(nn.Module):
    def __init__(self, input_dims, latent_dim):
        super(MVAE, self).__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        
        # Encoder networks for each modality
        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        ) for dim in input_dims])
        
        # Multi-modal attention module
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * len(input_dims), latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, len(input_dims))
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dims),
            nn.Sigmoid()
        )
        
    def forward(self, modalities):
        # Encode each modality
        encoded_modalities = [encoder(m) for m, encoder in zip(modalities, self.encoders)]
        
        # Compute attention weights
        flattened_modalities = torch.cat(encoded_modalities, dim=1)
        attention_weights = F.softmax(self.attention(flattened_modalities), dim=1)
        
        # Compute weighted sum of modalities
        weighted_sum = torch.zeros_like(encoded_modalities[0])
        for i, attention_weight in enumerate(attention_weights.split(1, dim=1)):
            weighted_sum += attention_weight * encoded_modalities[i]
        
        # Decode latent representation
        decoded = self.decoder(weighted_sum)
        
        return decoded, attention_weights
