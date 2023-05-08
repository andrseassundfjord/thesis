import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class TimeBERT(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout, embedding_dim = 4, cat_cols = None):
        super(TimeBERT, self).__init__()

        seq_len = input_dims[1][0]
        self.hidden_dim = hidden_layers[1]
        self.num_hidden_layers = hidden_layers[2]
        self.latent_dim = latent_dim
        self.mask_value = -999
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
            else:
                encoded_mpeds = tuple(torch.zeros_like(tensor) for tensor in encoded_standard)
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
            else:
                encoded_mcars = tuple(torch.zeros_like(tensor) for tensor in encoded_standard)

        zs, mus, logvars, encoded_features = zip(*[encoded_standard, encoded_mpeds, encoded_mcars])
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

class TimeseriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout, num_layers=1, categorical_cols=None, embedding_dim = 16, seq_len = 200):
        super(TimeseriesEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.categorical_cols = categorical_cols

        transformer_layers = 2
        self.transformer_d_model = hidden_dim
        transformer_num_heads = 8
        transformer_dff = 2048
        transformer_dropout = dropout
        
        if categorical_cols is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=num_cardinals, embedding_dim=embedding_dim)
                                             for num_cardinals in categorical_cols])
            self.input_dim += embedding_dim * len(self.categorical_cols)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model= self.transformer_d_model, 
                nhead=transformer_num_heads, 
                dim_feedforward=transformer_dff, 
                dropout=transformer_dropout,
                batch_first=True
            ), 
            num_layers=transformer_layers
        )
        
        self.fc1 = nn.Linear(hidden_dim * seq_len, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, cat_inp1 = None, cat_inp2 = None):
        """
        x: continuous features
        cat_inp1: categorical variables with 1 and 0 as possible values
        cat_inp2: categorical variables with more than two possible values
        """
        if x.size(2) == 1:
            print("Unexpected value in TimeseriesVAE encoder input: ", x.size(), ", expected: ", self.input_dim)
            zeros = torch.zeros(x.size(0), self.latent_dim).to("cuda" if torch.cuda.is_available() else "cpu")
            return zeros, zeros, zeros
        if self.categorical_cols is not None:
            cat_inputs = []
            for i, num_cardinals in enumerate(self.categorical_cols):
                if len(cat_inp2.size()) == 3:
                    cat_input = cat_inp2[:, :, i].long()
                else: 
                    cat_input = cat_inp2.long()
                if torch.max(cat_input).item() >= self.categorical_cols[i] or torch.min(cat_input).item() < 0:
                    print("Max value is greater or equal to embedding size, components.py line 168")
                    print(f"Max value: {torch.max(cat_input).item()}, Min value: {torch.min(cat_input).item()}, embedding size: {self.categorical_cols[i]}")
                    print(f"Number of categorical2 features: {len(self.categorical_cols)}, i: {i}", flush = True)
                cat_inputs.append(self.embeddings[i](cat_input))
            cat_inputs = torch.cat(cat_inputs, dim=-1)
            if cat_inp1 is not None:
                if len(cat_inp1.size()) < 3: 
                    cat_inp1 = cat_inp1.unsqueeze(-1)
                x = torch.cat([x, cat_inp1, cat_inputs], dim=-1)
            else:
                x = torch.cat([x, cat_inputs], dim=-1)
        encoded_features = self.transformer_encoder(x)
        features = encoded_features.view(x.size(0), -1)
        z = F.leaky_relu(self.fc1(features))
        z_mean = self.fc_mean(z)
        z_logvar = self.fc_logvar(z)
        return self.sampling((z_mean, z_logvar)), z_mean, z_logvar, encoded_features
    
    def sampling(self, args):
        mu, log_var = args
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class TimeseriesDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_shape, dropout = 0.1, num_layers=1, categorical_cols=None, embedding_dim=16):
        super(TimeseriesDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.categorical_cols = categorical_cols
        self.embedding_dim = embedding_dim
        seq_len = output_shape[0]

        transformer_layers = 2
        transformer_num_heads = 8
        transformer_dff = 2048
        
        # Add a transformer decoder layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, 
                                                                    nhead=transformer_num_heads,
                                                                    dim_feedforward=transformer_dff,
                                                                    dropout=dropout,
                                                                    batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=transformer_layers)

        if self.categorical_cols is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(num_cardinals, self.embedding_dim) for num_cardinals in self.categorical_cols])
            self.input_dim = self.latent_dim + self.embedding_dim * len(self.categorical_cols)
        else:
            self.input_dim = self.latent_dim
        # Define layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_shape[1])

        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        for name, param in self.transformer_decoder.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    init.kaiming_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

    def forward(self, z, categorical_input, memory):
        batch_size = z.size(0)
        seq_len = self.output_shape[0]
        if categorical_input is not None:
            categorical_embeddings = []
            for i, num_cardinals in enumerate(self.categorical_cols):
                if len(categorical_input.size()) == 3:
                    inp = categorical_input[:, :, i].long()
                else:
                    inp = categorical_input.long()
                if torch.max(inp).item() >= self.categorical_cols[i] or torch.min(inp).item() < 0:
                    print("Max value is greater or equal to embedding size, components.py line 168")
                    print(f"Max value: {torch.max(inp).item()}, Min value: {torch.min(inp).item()}, embedding size: {self.categorical_cols[i]}")
                    print(f"Number of categorical2 features: {len(self.categorical_cols)}, i: {i}", flush = True)
                categorical_embeddings.append(self.embeddings[i](inp))
            categorical_embeddings = torch.cat(categorical_embeddings, dim = -1)
            z = z.view(batch_size, 1, z.size(1))
            z = z.repeat(1, seq_len, 1)
            z = torch.cat([z, categorical_embeddings], dim=-1)
        
        # Apply the first linear layer to obtain the transformer input
        transformer_input = F.leaky_relu(self.fc1(z))  # shape: (batch_size, transformer_d_model)
        # Expand the transformer input to have the same length as the predicted frames
        transformer_input = transformer_input.unsqueeze(1).expand(-1, self.num_predicted_frames, -1)  # shape: (batch_size, num_predicted_frames, transformer_d_model)
        # Generate a mask tensor to prevent the decoder from attending to future frames
        mask = torch.ones(self.num_predicted_frames, self.num_predicted_frames).to(self.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        mask = mask.repeat(input.size(0) * 8, 1, 1)
        # Apply the transformer decoder to generate the predicted future frames
        decoded_frames = self.transformer_decoder(transformer_input, memory, tgt_mask=mask)
        output = F.relu(self.fc2(decoded_frames))
        return output
    
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)