import torch
import torch.nn as nn
import torch.nn.functional as F
from components import TimeseriesEncoder, TimeseriesDecoder
#import sys
#sys.path.append('../pyGAT')
#from models import GAT

class TimeseriesVAE(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout, embedding_dim = 4, cat_cols = None):
        super(TimeseriesVAE, self).__init__()

        seq_len = input_dims[1][0]
        self.hidden_dim = hidden_layers[1]
        self.num_hidden_layers = hidden_layers[2]
        self.latent_dim = latent_dim
        # Encoder
        self.standard_encoder = TimeseriesEncoder(
            input_dim = 35, # cont_shape + cat1_shape
            hidden_dim = self.hidden_dim,
            latent_dim = latent_dim,
            num_layers = self.num_hidden_layers,
            categorical_cols = [11, 64, 11, 7, 7, 7, 3, 3, 6, 4, 4, 7], # From features_used.txt
            embedding_dim = embedding_dim,
            dropout=dropout,
            seq_len = seq_len
        )
        # Decoder
        self.standard_decoder = TimeseriesDecoder(
            latent_dim = latent_dim, 
            hidden_dim = self.hidden_dim, 
            output_shape = (seq_len, 47), # cont_shape + cat1_shape + cat2_shape
            num_layers = self.num_hidden_layers,
            categorical_cols = [11, 64, 11, 7, 7, 7, 3, 3, 6, 4, 4, 7],
            embedding_dim = embedding_dim,
            dropout=dropout
        )
        # Mobileye car encoder
        self.mcars_encoder = TimeseriesEncoder(
            input_dim = 10, # cont_shape + cat1_shape
            hidden_dim = self.hidden_dim,
            latent_dim = latent_dim,
            num_layers = self.num_hidden_layers,
            categorical_cols = [5, 5, 6, 6, 4], # From features_used.txt
            embedding_dim = embedding_dim,
            dropout=dropout,
            seq_len = seq_len
        )
        # Mobileye car decoder
        self.mcars_decoder = TimeseriesDecoder(
            latent_dim = latent_dim, 
            hidden_dim = self.hidden_dim, 
            output_shape = (seq_len, 15), # cont_shape + cat1_shape + cat2_shape
            num_layers = self.num_hidden_layers,
            categorical_cols = [5, 5, 6, 6, 4],
            embedding_dim = embedding_dim,
            dropout=dropout
        )
        # Mobileye pedestrians encoder
        self.mpeds_encoder = TimeseriesEncoder(
            input_dim = 4, # cont_shape + cat1_shape
            hidden_dim = self.hidden_dim,
            latent_dim = latent_dim,
            num_layers = self.num_hidden_layers,
            categorical_cols = [3], # From features_used.txt
            embedding_dim = embedding_dim,
            dropout=dropout,
            seq_len = seq_len
        )
        # Mobileye pedestrians decoder
        self.mpeds_decoder = TimeseriesDecoder(
            latent_dim = latent_dim, 
            hidden_dim = self.hidden_dim, 
            output_shape = (seq_len, 5), # cont_shape + cat1_shape + cat2_shape
            num_layers = self.num_hidden_layers,
            categorical_cols = [3],
            embedding_dim = embedding_dim,
            dropout=dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3)
        )
        
    def forward(self, input):
        # Encode standard input 
        encoded_standard = self.standard_encoder(input[0], input[1], input[2])
        # Mobileye pedestrians
        encoded_mpeds = []
        if input[3] is not None and input[4].size(2) > 0:
            if input[3].size(2) != input[4].size(2) * 4:
                print("Error with mobileye pedestrians shape, add cols")
                print(f"Input 3: {input[3].size(2)}")
                print(f"Input 4: {input[4].size(2)}, should 2 x input 3", flush = True)
            for i in range(input[4].size(2)):
                peds_cont = input[3][:,:,4*i:4*i+4]
                peds_cat2 = input[4][:,:,i]
                encoded_mpeds.append(self.mpeds_encoder(peds_cont, cat_inp2 = peds_cat2))
            if len(encoded_mpeds) > 0:
                encoded_mpeds = tuple(torch.sum(torch.stack(tensors), dim=0) for tensors in zip(*encoded_mpeds))
                #kl_divergence += -0.5 * torch.sum(1 + encoded_mpeds[2] - encoded_mpeds[1].pow(2) - encoded_mpeds[2].exp(), dim=1).mean()
                #decoded_mpeds = [self.mpeds_decoder(encoded_mpeds[0], input[4]) for _ in range(input[4].size(2))]
                #decoded_mpeds = torch.cat(decoded_mpeds, dim = -1)
            else:
                encoded_mpeds = tuple(torch.zeros_like(tensor) for tensor in encoded_standard)
                #decoded_mpeds = torch.cat(input[3:5], dim = -1)
                #print(f"decoded mped cat size {decoded_mpeds.size()} in line 99 TimeSeriesVAE")
        # Mobileye cars
        encoded_mcars = []
        if input[5] is not None and input[6].size(2) > 0:
            if input[6].size(2) != int(input[5].size(2) / 9) or input[6].size(2) != int(input[7].size(2) / 5):
                print("Error with mobileye cars shape, add cols")
                print(f"Input 5: {input[5].size(2)}, should 9 x input 6")
                print(f"Input 6: {input[6].size(2)}")  
                print(f"Input 7: {input[7].size(2)}, should 5 x input 6", flush = True)                
            for i in range(input[6].size(2)):
                cars_cont = input[5][:,:,9*i:9*i+9]
                cars_cat1 = input[6][:,:,i]
                cars_cat2 = input[7][:,:,5*i:5*i+5]
                encoded_mcars.append(self.mcars_encoder(cars_cont, cars_cat1, cars_cat2))
            if len(encoded_mcars) > 0:
                encoded_mcars = tuple(torch.sum(torch.stack(tensors), dim=0) for tensors in zip(*encoded_mcars))
                #kl_divergence += -0.5 * torch.sum(1 + encoded_mcars[2] - encoded_mcars[1].pow(2) - encoded_mcars[2].exp(), dim=1).mean()
                #decoded_mcars = [self.mcars_decoder(encoded_mcars[0], input[7]) for _ in range(input[6].size(2))]
                #decoded_mcars = torch.cat(decoded_mcars, dim = -1)
            else:
                encoded_mcars = tuple(torch.zeros_like(tensor) for tensor in encoded_standard)
                #decoded_mcars = torch.cat(input[5:], dim = -1)

        #encoded = tuple(torch.sum(torch.stack(tensors), dim=0) for tensors in zip(*[encoded_standard, encoded_mpeds, encoded_mcars]))
        #kl_divergence = -0.5 * torch.sum(1 + encoded[2] - encoded[1].pow(2) - encoded[2].exp(), dim=1).mean()
        zs, mus, logvars = zip(*[encoded_standard, encoded_mpeds, encoded_mcars])
        kl_divergence = -0.5 * torch.sum(1 + torch.stack(logvars) - torch.stack(mus).pow(2) - torch.stack(logvars).exp(), dim=1).mean()
        
        flattened_modalities = torch.cat(zs, dim=1)
        attention_weights = F.softmax(self.attention(flattened_modalities), dim=1)
        
        # Compute weighted sum of modalities
        weighted_sum = torch.zeros_like(zs[0])
        for i, attention_weight in enumerate(attention_weights.split(1, dim=1)):
            weighted_sum += attention_weight * zs[i]

        # Compute attention weights
        flattened_mus = torch.cat(mus, dim=1)
        attention_weights_mus = F.softmax(self.attention(flattened_mus), dim=1)

        # Compute weighted mean of means
        weighted_mu = torch.zeros_like(mus[0])
        for i, attention_weight in enumerate(attention_weights_mus.split(1, dim=1)):
            weighted_mu += attention_weight * mus[i]


        decoded_standard = self.standard_decoder(weighted_sum, input[2])
        decoded_mpeds = [self.mpeds_decoder(weighted_sum, input[4]) for _ in range(input[4].size(2))]
        decoded_mpeds = torch.cat(decoded_mpeds, dim = -1)
        decoded_mcars = [self.mcars_decoder(weighted_sum, input[7]) for _ in range(input[6].size(2))]
        decoded_mcars = torch.cat(decoded_mcars, dim = -1)
        return (decoded_standard, decoded_mpeds, decoded_mcars), kl_divergence, weighted_sum, weighted_mu


