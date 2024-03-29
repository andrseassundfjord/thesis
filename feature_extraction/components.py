import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models

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
    def __init__(self, latent_dim, input_shape, hidden_shape, dropout, hidden_dim = 512):
        super(VideoEncoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]
        self.hs = hidden_shape

        self.model = nn.ModuleList([
            nn.Conv3d(self.n_channel, self.hs[0], kernel_size = 4, stride = 2, padding = 1),
            #nn.BatchNorm3d(self.hs[0]),
            nn.LeakyReLU(),
            nn.Conv3d(self.hs[0], self.hs[1], kernel_size = 4, stride = 2, padding = 1),
            #nn.BatchNorm3d(self.hs[1]),
            nn.LeakyReLU(),
            nn.Conv3d(self.hs[1], self.hs[2], kernel_size = 4, stride = 2, padding = 1),
            #nn.BatchNorm3d(self.hs[2]),
            nn.LeakyReLU(),
            nn.Conv3d(self.hs[2], self.hs[3], kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(),
            #nn.MaxPool3d(kernel_size = (3, 4, 4), stride = (1, 2, 2), padding = 1),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(int(self.hs[3] * self.width * self.height * self.n_frames / (16 * 16 * 16)), hidden_dim),
            #nn.BatchNorm1d(self.hs[3]),
            nn.LeakyReLU()
        ])
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                #init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        init.kaiming_normal_(self.fc_mu.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.fc_log_var.weight, mode='fan_in', nonlinearity='leaky_relu')


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

class VideoEncoderPretrained(nn.Module):
    def __init__(self, latent_dim, input_shape, hidden_shape, dropout, hidden_dim = 512):
        super(VideoEncoderPretrained, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]
        self.hs = hidden_shape

        self.backbone = models.video.r3d_18()

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(512, self.hs[-1])

        self.fc_mu = nn.Linear(self.hs[-1], latent_dim)
        self.fc_log_var = nn.Linear(self.hs[-1], latent_dim)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        if torch.any(torch.isnan(x)):
            print("NaN in: Encoder input")
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class VideoDecoder(nn.Module):
    def __init__(self, latent_dim, input_shape, hidden_shape, dropout, hidden_dim = 512):
        super(VideoDecoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]
        self.hs = hidden_shape

        self.model = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim),
            #nn.BatchNorm1d(self.hs[3]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(self.hs[3] * self.width * self.height * self.n_frames / (16 ** 3))),
            #nn.BatchNorm1d(int(self.hs[2] * self.width * self.height * self.n_frames / (8 ** 3))),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            Reshape((self.hs[3], int(self.n_frames / 16), int(self.width / 16), int(self.height / 16))),
            nn.ConvTranspose3d(self.hs[3], self.hs[2], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(self.hs[2], self.hs[1], kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm3d(self.hs[1]),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(self.hs[1], self.hs[0], kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm3d(self.hs[0]),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(self.hs[0], self.n_channel, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm3d(self.n_channel),
            nn.Sigmoid()
        ])

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.ConvTranspose3d, nn.Linear)):
                #init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
            if torch.any(torch.isnan(x)):
                print("NaN in: Decoder ", layer)
            #print(layer, x.size())
        return x

class TimeseriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout, num_layers=1, categorical_cols=None, embedding_dim = 16, seq_len = 200):
        super(TimeseriesEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.categorical_cols = categorical_cols
        self.bidirectional = False
        self.embedding_dim = embedding_dim
        self.mask_value = -99
        
        if categorical_cols is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=num_cardinals, embedding_dim=embedding_dim)
                                             for num_cardinals in categorical_cols])
            self.input_dim += embedding_dim * len(self.categorical_cols)

        #self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers=num_layers, batch_first = True, dropout = dropout, bidirectional = self.bidirectional)
        self.gru = nn.GRU(self.input_dim, hidden_dim, num_layers = num_layers, batch_first = True, dropout = dropout, bidirectional = self.bidirectional)
        self.flatten = Flatten()
        bidir = 2 if self.bidirectional else 1
        self.fc1 = nn.Linear(hidden_dim * seq_len * bidir, hidden_dim * 2)
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        # Define batchnorm
        self.bn = nn.BatchNorm1d(seq_len)

        self.leaky = nn.LeakyReLU()

        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')

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
                x = torch.cat([x, cat_inp1, cat_inputs], dim=-1)
            else:
                x = torch.cat([x, cat_inputs], dim=-1)
        #lstm_out, _ = self.lstm(x)
        lstm_out, _ = self.gru(x)
        lstm_out = self.bn(lstm_out)
        lstm_out = self.flatten(lstm_out)
        z = self.leaky(self.fc1(lstm_out))
        z_mean = self.fc_mean(z)
        z_logvar = self.fc_logvar(z)
        return self.sampling((z_mean, z_logvar)), z_mean, z_logvar
    
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
        self.bidirectional = False
        seq_len = output_shape[0]
        
        #if self.categorical_cols is not None:
        #    self.embeddings = nn.ModuleList([nn.Embedding(num_cardinals, self.embedding_dim) for num_cardinals in self.categorical_cols])
        #    self.input_dim = self.latent_dim + self.embedding_dim * len(self.categorical_cols)
        #else:
        self.input_dim = self.latent_dim
        # Define layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout, bidirectional = self.bidirectional)
        bidir = 2 if self.bidirectional else 1
        self.fc3 = nn.Linear(hidden_dim * bidir, output_shape[1])
        # Define batchnorm
        self.bn = nn.BatchNorm1d(seq_len)

        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, z, categorical_input=None):
        batch_size = z.size(0)
        seq_len = self.output_shape[0]
        """
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
            z = torch.cat([z, categorical_embeddings], dim=-1)
        """
        z = z.view(batch_size, 1, z.size(1))
        z = z.repeat(1, seq_len, 1)
        z = self.leaky(self.fc1(z))
        lstm_out, _ = self.gru(z)
        lstm_out = self.bn(lstm_out)
        output = self.relu(self.fc3(lstm_out))
        return output

class TimeseriesEncoder2(nn.Module):
    def __init__(self, hidden_dim, latent_dim, dropout, num_layers=1, categorical=None, embedding_dim = 4, bidirectional = False):
        super(TimeseriesEncoder2, self).__init__()
        self.input_dim = 1
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.categorical = categorical
        self.seq_len = 200
        
        #if self.categorical is not None:
        #    self.embedding = nn.Embedding(num_embeddings=categorical, embedding_dim=embedding_dim)
        #    self.input_dim = embedding_dim
        self.lstm = nn.LSTM(self.input_dim * self.seq_len, hidden_dim, num_layers=num_layers, batch_first = True, dropout = dropout, bidirectional = self.bidirectional)
        """for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)"""
        #self.gru = nn.GRU(self.input_dim, hidden_dim, num_layers = num_layers, batch_first = True, dropout = dropout, bidirectional = self.bidirectional)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        # Define batchnorm
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """
        x: continuous features
        """
        if self.categorical is not None:
            x = self.embedding(x)
            print("Embed", flush = True)
        x = self.flatten(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.bn(lstm_out)
        #lstm_out = self.flatten(lstm_out)
        z = F.relu(self.fc1(lstm_out))
        z_mean = self.fc_mean(z)
        z_logvar = self.fc_logvar(z)
        return self.sampling((z_mean, z_logvar)), z_mean, z_logvar
    
    def sampling(self, args):
        mu, log_var = args
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class TimeseriesDecoder2(nn.Module):
    def __init__(self, latent_dim, hidden_dim, dropout = 0.1, num_layers=1, categorical=None, embedding_dim=4, bidirectional = False):
        super(TimeseriesDecoder2, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = 200 if categorical == None else embedding_dim
        self.num_layers = num_layers
        self.categorical = categorical
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.input_dim = latent_dim
        
        #"if self.categorical is not None:
        #    self.embedding = nn.functional.embedding(indices, embedding_matrix)
        # Define layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout, bidirectional = self.bidirectional)
        self.fc3 = nn.Linear(hidden_dim, self.output_size)
        # Define batchnorm
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        lstm_out, _ = self.lstm(z)
        lstm_out = self.bn(lstm_out)

        output = self.fc3(lstm_out).unsqueeze(-1)
        # if categorical is not None:
        # reverse embed from 4 columns to one
        return output

