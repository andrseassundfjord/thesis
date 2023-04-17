import torch
import torch.nn as nn
import torch.nn.functional as F
from components import TimeseriesEncoder2, TimeseriesDecoder2

class TimeseriesVAE2(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout, embedding_dim = 8):
        super(TimeseriesVAE2, self).__init__()

        self.hidden_dim = hidden_layers[1]
        self.num_hidden_layers = hidden_layers[2]
        self.latent_dim = latent_dim
        self.num_features = 67
        self.cat_cols = [11, 64, 11, 7, 7, 7, 3, 3, 6, 4, 4, 7, 3, 5, 5, 6, 6, 4]
        self.cat_cols_idx = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 62, 63, 64, 65, 66]
        self.feature_len_list = [29, 6, 12, 4, 1, 9, 1, 5]
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        bidirectional = False

        counter = 0
        for i in range(self.num_features):
            categorical = None
            if i == self.cat_cols_idx[counter]:
                categorical = self.cat_cols[counter]
                counter += 1
            # Encoders
            encoder = TimeseriesEncoder2(
                hidden_dim = self.hidden_dim, 
                latent_dim = self.latent_dim,
                num_layers = self.num_hidden_layers,
                categorical = categorical,
                embedding_dim = embedding_dim,
                dropout = dropout, 
                bidirectional = bidirectional
            )
            self.encoder_list.append(encoder)
            # Decoder
            decoder = TimeseriesDecoder2(
                latent_dim = latent_dim, 
                hidden_dim = self.hidden_dim, 
                num_layers = self.num_hidden_layers,
                categorical = categorical,
                embedding_dim = embedding_dim,
                dropout=dropout, 
                bidirectional = bidirectional
            )
            self.decoder_list.append(decoder)        

    def forward(self, input):
        # Encode standard input
        encoded_list = []
        for input_idx, input_type in enumerate(input):
            for feature_idx in range(input_type.size(2)):
                idx = feature_idx
                while idx >= self.feature_len_list[input_idx]:
                    idx -= self.feature_len_list[input_idx]
                feature = input_type[:, :, feature_idx].unsqueeze(-1)
                encoded = self.encoder_list[idx](feature)
                encoded_list.append(encoded)

        nan_mask = torch.isnan(encoded_list[0][0])
        if torch.any(nan_mask):
            print("nan in encoded_list[0]", flush = True)
        
        encoded = tuple(torch.sum(torch.stack(tensors), dim=0) for tensors in zip(*encoded_list))
        kl_divergence = -0.5 * torch.sum(1 + encoded[2] - encoded[1].pow(2) - encoded[2].exp(), dim=1).mean()
        decoded_list = []
        idx = 0
        for input_idx, input_type in enumerate(input):
            for feature_idx in range(input_type.size(2)):
                if idx >= self.feature_len_list[input_idx]:
                    idx -= self.feature_len_list[input_idx]
                decoded = self.decoder_list[idx](encoded[0])
                decoded_list.append(decoded)
                idx += 1
                
        decoded_cat = torch.cat(decoded_list, dim = -1)
        nan_mask = torch.isnan(decoded_cat[0])
        if torch.any(nan_mask):
            print("nan in decoded_cat", flush = True)

        return decoded_cat, kl_divergence


