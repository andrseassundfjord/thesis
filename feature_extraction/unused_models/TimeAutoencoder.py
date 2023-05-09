import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

class TimeseriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout, num_layers=1, categorical_cols=None, embedding_dim = 16):
        super(TimeseriesEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.categorical_cols = categorical_cols
        self.bidirectional = False
        
        if categorical_cols is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=num_cardinals, embedding_dim=embedding_dim)
                                             for num_cardinals in categorical_cols])
            self.input_dim += embedding_dim * len(self.categorical_cols)

        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers=num_layers, batch_first = True, dropout = dropout, bidirectional = self.bidirectional)
        #self.gru = nn.GRU(self.input_dim, hidden_dim, num_layers = num_layers, batch_first = True, dropout = dropout, bidirectional = self.bidirectional)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(hidden_dim * 200, hidden_dim * 2)
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        # Define batchnorm
        self.bn = nn.BatchNorm1d(200)

    def forward(self, x, cat_inp1 = None, cat_inp2 = None):
        """
        x: continuous features
        cat_inp1: categorical variables with 1 and 0 as possible values
        cat_inp2: categorical variables with more than two possible values
        """
        
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
        #lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out, _ = self.lstm(x)
        lstm_out = self.bn(lstm_out)
        lstm_out = self.flatten(lstm_out)
        z = F.relu(self.fc1(lstm_out))
        z_mean = self.fc_mean(z)
        z_logvar = self.fc_logvar(z)
        return self.sampling((z_mean, z_logvar)), z_mean, z_logvar

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
        
        if self.categorical_cols is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(num_cardinals, self.embedding_dim) for num_cardinals in self.categorical_cols])
            self.input_dim = self.latent_dim + self.embedding_dim * len(self.categorical_cols)
        else:
            self.input_dim = self.latent_dim
        # Define layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout = dropout, bidirectional = self.bidirectional)
        self.fc3 = nn.Linear(hidden_dim, output_shape[1])
        # Define batchnorm
        self.bn = nn.BatchNorm1d(200)

    def forward(self, z, categorical_input=None):
        batch_size = z.size(0)
        seq_len = self.output_shape[0]
        bidir = 2 if self.bidirectional else 1
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
        
        z = F.relu(self.fc1(z))
        lstm_out, _ = self.lstm(z)
        lstm_out = self.bn(lstm_out)
        output = self.fc3(lstm_out)
        return output