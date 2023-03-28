import torch
import torch.nn as nn
import torch.nn.functional as F
from components import TimeseriesEncoder, TimeseriesDecoder

class TimeseriesVAE(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout, embedding_dim = 16, cat_cols = None):
        super(TimeseriesVAE, self).__init__()

        self.hidden_dim = hidden_layers[1]
        self.num_hidden_layers = hidden_layers[2]
        self.latent_dim = latent_dim
        # Encoder
        self.standard_encoder = TimeseriesEncoder(
            input_dim = 36, # cont_shape + cat1_shape
            hidden_dim = self.hidden_dim,
            latent_dim = latent_dim,
            num_layers = self.num_hidden_layers,
            categorical_cols = [11, 63, 10, 3, 7, 7, 3, 3, 6, 4, 4], # From features_used.txt
            embedding_dim = embedding_dim,
            dropout=dropout
        )
        # Decoder
        self.standard_decoder = TimeseriesDecoder(
            latent_dim = latent_dim, 
            hidden_dim = self.hidden_dim, 
            output_shape = (200, 36), # cont_shape + cat1_shape
            num_layers = self.num_hidden_layers,
            categorical_cols = [11, 63, 10, 3, 7, 7, 3, 3, 6, 4, 4],
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
            dropout=dropout
        )
        # Mobileye car decoder
        self.mcars_decoder = TimeseriesDecoder(
            latent_dim = latent_dim, 
            hidden_dim = self.hidden_dim, 
            output_shape = (200, 10), # cont_shape + cat1_shape
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
            categorical_cols = [3, 5], # From features_used.txt
            embedding_dim = embedding_dim,
            dropout=dropout
        )
        # Mobileye pedestrians decoder
        self.mpeds_decoder = TimeseriesDecoder(
            latent_dim = latent_dim, 
            hidden_dim = self.hidden_dim, 
            output_shape = (200, 4), # cont_shape + cat1_shape
            num_layers = self.num_hidden_layers,
            categorical_cols = [3, 5],
            embedding_dim = embedding_dim,
            dropout=dropout
        )
        

    def forward(self, input):
        # Encode standard input
        encoded_standard = self.standard_encoder(input[0], input[1], input[2])
        # Mobileye pedestrians
        if input[3].size(1) != (input[4].size(1) / 2):
            print("Error with mobileye pedestrians shape, add cols")
        encoded_mpeds = torch.zeros(self.latent_dim)
        for i in range(input[3].size(1) / 2):
            peds_cont = input[3][:,:,2*i:2*(i+2)]
            peds_cat2 = input[4][:,:,4*i:4*(i+4)]
            encoded_mpeds += self.mpeds_encoder(peds_cont, peds_cat2)
        # Mobileye cars
        if input[6].size(1) != (input[5].size(1) / 9):
            print("Error with mobileye cars shape, add cols")
        if input[6].size(1) != (input[7].size(1) / 5):
            print("Error with mobileye cars shape, add cols")
        encoded_mcars = torch.zeros(self.latent_dim)
        for i in range(input[6].size(1)):
            cars_cont = input[5][:,:,9*i:9*(i+9)]
            cars_cat1 = input[6][:,:,i]
            cars_cat2 = input[7][:,:,5*i:5*(i+5)]
            encoded_mcars += self.mcars_encoder(cars_cont, cars_cat1, cars_cat2)

        kl_divergence = -0.5 * torch.sum(1 + encoded_standard[2] - encoded_standard[1].pow(2) - encoded_standard[2].exp(), dim=1).mean()
        kl_divergence += -0.5 * torch.sum(1 + encoded_mpeds[2] - encoded_mpeds[1].pow(2) - encoded_mpeds[2].exp(), dim=1).mean()
        kl_divergence += -0.5 * torch.sum(1 + encoded_mcars[2] - encoded_mcars[1].pow(2) - encoded_mcars[2].exp(), dim=1).mean()

        decoded_standard = self.standard_decoder(encoded_standard[0], input[2])
        decoded_mpeds = self.mpeds_decoder(encoded_mpeds[0], input[4])
        decoded_mcars = self.mcars_decoder(encoded_mcars[0], input[7])
        return (decoded_standard, decoded_mpeds, decoded_mcars), kl_divergence

