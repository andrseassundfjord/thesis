import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models

class MidMVAE(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout):
        super(MidMVAE, self).__init__()

        self.video_input_shape = input_dims[0]
        self.timeseries_input_shape = input_dims[1]
        self.hidden_shape = hidden_layers[0]
        self.hidden_dim = hidden_layers[1]
        self.num_hidden_layers = hidden_layers[2]
        embedding_dim = 4
        seq_len = input_dims[1][0]
        new_seq_len = input_dims[0][0]
        # Video
        self.video_encoder = VideoEncoder(latent_dim = latent_dim, 
                                            input_shape = self.video_input_shape,
                                            cnn_filters=hidden_layers[0],
                                            hidden_dim = self.hidden_dim,
                                            dropout = dropout
                                        )
        self.video_decoder = VideoDecoder(latent_dim = latent_dim, 
                                            input_shape = self.video_input_shape, 
                                            hidden_shape = self.hidden_shape,
                                            hidden_dim = self.hidden_dim * 2,
                                            dropout = dropout
                                        )
        # Timeseries
        self.standard_encoder = TimeseriesEncoder(
            input_dim = 35, # cont_shape + cat1_shape
            hidden_dim = self.hidden_dim,
            categorical_cols = [11, 64, 11, 7, 7, 7, 3, 3, 6, 4, 4, 7], # From features_used.txt
            embedding_dim = embedding_dim,
            dropout=dropout,
            seq_len=seq_len,
            new_seq_len=new_seq_len
        )
        # Decoder
        self.standard_decoder = TimeseriesDecoder(input_dim = self.hidden_dim * 2, output_shape = (seq_len, 47), dropout = dropout, old_seq_len = new_seq_len)
        # Mobileye car encoder
        self.mcars_encoder = TimeseriesEncoder(
            input_dim = 10, # cont_shape + cat1_shape
            hidden_dim = self.hidden_dim,
            categorical_cols = [5, 5, 6, 6, 4], # From features_used.txt
            embedding_dim = embedding_dim,
            dropout=dropout,
            seq_len=seq_len,
            new_seq_len=new_seq_len
        )
        # Mobileye car decoder
        self.mcars_decoder = TimeseriesDecoder(input_dim = self.hidden_dim * 2, output_shape = (seq_len, 15), dropout = dropout, old_seq_len = new_seq_len)
        # Mobileye pedestrians encoder
        self.mpeds_encoder = TimeseriesEncoder(
            input_dim = 4, # cont_shape + cat1_shape
            hidden_dim = self.hidden_dim,
            categorical_cols = [3], # From features_used.txt
            embedding_dim = embedding_dim,
            dropout=dropout,
            seq_len=seq_len,
            new_seq_len=new_seq_len
        )
        # Mobileye pedestrians decoder
        self.mpeds_decoder = TimeseriesDecoder(input_dim = self.hidden_dim * 2, output_shape = (seq_len, 5), dropout = dropout, old_seq_len = new_seq_len)
        # Multi-modal attention module
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 3)
        )
        for m in self.attention:
            if isinstance(m, (nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        self.t_encoder = Encoder(self.hidden_dim * 2, hidden_dim=self.hidden_dim, latent_dim=latent_dim, dropout=dropout, num_frames=new_seq_len)
        self.t_decoder = Decoder(input_dim=latent_dim, hidden_dim=self.hidden_dim * 2, dropout=dropout, num_predicted_frames=new_seq_len)

    def encodeTimeseries(self, input):
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
                encoded_mpeds = torch.sum(torch.stack(encoded_mpeds), dim=0)
            else:
                encoded_mpeds = torch.zeros_like(encoded_standard)
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
                encoded_mcars = torch.sum(torch.stack(encoded_mcars), dim=0)
            else:
                encoded_mcars = torch.zeros_like(encoded_standard)

        return [encoded_standard, encoded_mpeds, encoded_standard]

    def encodeVideo(self, input):
        return [self.video_encoder(input)]

    def decodeTimeseries(self, weighted_sum, input):
        decoded_standard = self.standard_decoder(weighted_sum)
        decoded_mpeds = [self.mpeds_decoder(weighted_sum) for _ in range(input[4].size(2))]
        decoded_mpeds = torch.cat(decoded_mpeds, dim = -1)
        decoded_mcars = [self.mcars_decoder(weighted_sum) for _ in range(input[6].size(2))]
        decoded_mcars = torch.cat(decoded_mcars, dim = -1)
        return (decoded_standard, decoded_mpeds, decoded_mcars)
    
    def decodeVideo(self, weighted_sum):
        return self.video_decoder(weighted_sum)

    def sampling(self, args):
        mu, log_var = args
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, modalities):
        # Encode each modality
        encoded_video = self.encodeVideo(modalities[0])
        encoded_timeseries = self.encodeTimeseries(modalities[1])
        # Compute attention weights
        flattened_timeseries = torch.cat(encoded_timeseries, dim=-1)
        attention_weights = F.softmax(self.attention(flattened_timeseries), dim=-1)
        # Compute weighted sum of timeseries
        weighted_timeseries = torch.zeros_like(encoded_timeseries[0])
        for i, attention_weight in enumerate(attention_weights.split(1, dim=-1)):
            weighted_timeseries += attention_weight * encoded_timeseries[i]
        
        cat_encoded = torch.cat(encoded_video + [weighted_timeseries], dim = -1)
        # Unpack the encoded tensors and compute the Kullback-Leibler divergence loss
        mu, logvar, encoded_features = self.t_encoder(cat_encoded)
        z = self.sampling((mu, logvar))
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        decoded = self.t_decoder(z, encoded_features)

        # Decode latent representation
        decoded_video = self.decodeVideo(decoded)
        decoded_timeseries = self.decodeTimeseries(decoded, modalities[1])
        return decoded_video, decoded_timeseries, kl_divergence, z, mu

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout, num_frames):
        super(Encoder, self).__init__()
        transformer_layers = 2
        self.transformer_d_model = input_dim
        transformer_num_heads = 8
        transformer_dff = 2048
        transformer_dropout = dropout
                
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

        for name, param in self.transformer_encoder.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    init.kaiming_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

        self.fc_middle = nn.Linear(input_dim * num_frames, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        init.kaiming_normal_(self.fc_middle.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        special_token = torch.zeros((x.size(0), 1, x.size(2)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        special_token = torch.sub(special_token, -99).to(device)
        x = torch.cat([special_token, x], dim = 1)
        encoded_features = self.transformer_encoder(x)[:, 1:, :]
        # Take the last encoded frame as the representation of the input video
        encoded = encoded_features.view(x.size(0), -1)
        encoded = F.leaky_relu(self.fc_middle(encoded))
        # Apply the final linear layer to obtain the latent representation
        mu = self.fc_mu(encoded)  # shape: (batch_size, latent_dim)
        log_var = self.fc_log_var(encoded)
        return mu, log_var, encoded_features

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_predicted_frames):
        super(Decoder, self).__init__()
        self.num_predicted_frames = num_predicted_frames
        transformer_layers = 2
        transformer_num_heads = 8
        transformer_dff = 2048

        # Add a linear layer to map the latent representation to the transformer input
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Weight init for linear layers
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, 
                                                                    nhead=transformer_num_heads,
                                                                    dim_feedforward=transformer_dff,
                                                                    dropout=dropout,
                                                                    batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=transformer_layers)

        for name, param in self.transformer_decoder.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    init.kaiming_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

    def forward(self, input, memory):
        # Apply the first linear layer to obtain the transformer input
        transformer_input = F.leaky_relu(self.fc1(input))  # shape: (batch_size, transformer_d_model)
        # Expand the transformer input to have the same length as the predicted frames
        transformer_input = transformer_input.unsqueeze(1).expand(-1, self.num_predicted_frames, -1)  # shape: (batch_size, num_predicted_frames, transformer_d_model)
        # Generate a mask tensor to prevent the decoder from attending to future frames
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = torch.ones(self.num_predicted_frames, self.num_predicted_frames).to(device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        mask = mask.repeat(input.size(0) * 8, 1, 1)
        # Apply the transformer decoder to generate the predicted future frames
        decoded_frames = self.transformer_decoder(transformer_input, memory, tgt_mask=mask)
        return decoded_frames

class TimeseriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout = 0.1, categorical_cols=None, embedding_dim = 16, seq_len = 200, new_seq_len = 64):
        super(TimeseriesEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.categorical_cols = categorical_cols
        self.mask_value = -99
        self.embedding_dim = embedding_dim
        
        if categorical_cols is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=num_cardinals, embedding_dim=embedding_dim)
                                             for num_cardinals in categorical_cols])
            self.input_dim += embedding_dim * len(self.categorical_cols)

        self.model = nn.ModuleList([
            Permuter((0, 2, 1)),
            nn.Linear(seq_len, new_seq_len),
            #nn.BatchNorm1d(new_seq_len),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            Permuter((0, 2, 1)),
            nn.Linear(self.input_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        ])

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.Linear)):
                #init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)


    def forward(self, x, cat_inp1 = None, cat_inp2 = None):
        """
        x: continuous features
        cat_inp1: categorical variables with 1 and 0 as possible values
        cat_inp2: categorical variables with more than two possible values
        """
        if len(x.size()) == 2:
            print(f"Unexpected input dimensions: {x.size()}")
            zeros = torch.zeros(x.size(0), self.latent_dim).to("cuda" if torch.cuda.is_available() else "cpu")
            return zeros, zeros, zeros            
        elif x.size(2) == 1:
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
                if torch.max(cat_input).item() >= self.categorical_cols[i]:
                    print("Max value is greater or equal to embedding size, components.py line 168")
                    print(f"Max value: {torch.max(cat_input).item()}, Min value: {torch.min(cat_input).item()}, embedding size: {self.categorical_cols[i]}")
                    print(f"Number of categorical2 features: {len(self.categorical_cols)}, i: {i}", flush = True)
                if torch.min(cat_input).item() < 0:
                    mask = cat_input.eq(self.mask_value)
                    cat_input[mask] = 0.0
                    cat_input = self.embeddings[i](cat_input)
                    mask = mask.unsqueeze(2)
                    mask = torch.repeat_interleave(mask, self.embedding_dim, dim = -1)
                    cat_input[mask] = self.mask_value
                    cat_inputs.append(cat_input)
                else:
                    cat_inputs.append(self.embeddings[i](cat_input))
            cat_inputs = torch.cat(cat_inputs, dim=-1)
            if cat_inp1 is not None:
                if len(cat_inp1.size()) < 3: 
                    cat_inp1 = cat_inp1.unsqueeze(-1)
                x = torch.cat([x, cat_inp1, cat_inputs], dim=2)
            else:
                x = torch.cat([x, cat_inputs], dim=-1)
        
        for layer in self.model:
            x = layer(x)
            if torch.any(torch.isnan(x)):
                print("NaN in: Decoder ", layer)
        return x
    
class TimeseriesDecoder(nn.Module):
    def __init__(self, input_dim, output_shape, dropout = 0.1, old_seq_len = 64):
        super(TimeseriesDecoder, self).__init__()
        seq_len = output_shape[0]
        # Define layers
        self.model = nn.ModuleList([
            nn.Linear(input_dim, output_shape[1]),
            #nn.BatchNorm1d(output_shape[1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            Permuter((0, 2, 1)),
            nn.Linear(old_seq_len, seq_len),
            #nn.BatchNorm1d(seq_len),
            nn.ReLU(),
            nn.Dropout(dropout),
            Permuter((0, 2, 1))
        ])

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.Linear)):
                #init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, z):
        for layer in self.model:
            z = layer(z)
            if torch.any(torch.isnan(z)):
                print("NaN in: Decoder ", layer)
        return z
    
class VideoEncoder(nn.Module):
    def __init__(self, latent_dim, input_shape, cnn_filters, hidden_dim, dropout):
        super(VideoEncoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]

        kernel_size = (3, 4, 4)
        stride = (1, 2, 2)
        padding = (1, 1, 1)

        output_size = input_shape[1]
        for _ in range(len(cnn_filters)):
            output_size = int((output_size - kernel_size[1] + 2*padding[1])/stride[1]) + 1

        cnn_layers = []
        in_channels = self.n_channel
        # Define 3D CNN encoder
        for i in range(len(cnn_filters)):
            cnn_layers.append(nn.Conv3d(in_channels, cnn_filters[i], kernel_size = kernel_size, stride = stride, padding = padding))
            cnn_layers.append(nn.LeakyReLU(inplace=True))
            cnn_layers.append(nn.Dropout(dropout))
            #cnn_layers.append(nn.MaxPool3d(kernel_size = kernel_size, stride = stride, padding = padding))
            in_channels = cnn_filters[i]
        #cnn_layers.append(nn.MaxPool3d(kernel_size = (3, 4, 4), stride = (1, 2, 2), padding = 1))
        self.cnn_encoder = nn.Sequential(*cnn_layers)


        self.model = nn.ModuleList([
            nn.Linear(in_channels * output_size ** 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        ])

        for m in self.cnn_encoder:
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)


    def forward(self, x):
        cnn_encoded = self.cnn_encoder(x)
        features = cnn_encoded.view(x.size(0), x.size(2), -1)
        for layer in self.model:
            features = layer(features)
        return features

class VideoDecoder(nn.Module):
    def __init__(self, latent_dim, input_shape, hidden_shape, hidden_dim, dropout):
        super(VideoDecoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]
        self.hs = hidden_shape

        kernel_size = (3, 4, 4)
        stride = (1, 2, 2)
        padding = (1, 1, 1)

        output_size = input_shape[1]
        for _ in range(len(hidden_shape)):
            output_size = int((output_size - kernel_size[1] + 2*padding[1])/stride[1]) + 1

        self.model = nn.ModuleList([
            nn.Linear(hidden_dim, self.hs[3] * output_size ** 2),
            #nn.BatchNorm2d(self.hs[2] * output_size ** 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            Reshape((self.hs[3], self.n_frames, output_size, output_size)),
            nn.ConvTranspose3d(self.hs[3], self.hs[2], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(self.hs[2], self.hs[1], kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.BatchNorm3d(self.hs[1]),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(self.hs[1], self.hs[0], kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.BatchNorm3d(self.hs[0]),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(self.hs[0], self.n_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.BatchNorm3d(self.n_channel),
            nn.Sigmoid()
        ])

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.ConvTranspose3d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
            if torch.any(torch.isnan(x)):
                print("NaN in: Decoder ", layer)
        return x
    
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)
    
class Permuter(torch.nn.Module):
    def __init__(self, new_order):
        super(Permuter, self).__init__()
        self.new_order = new_order

    def forward(self, x):
        return torch.permute(x, self.new_order)