import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(GRUModel, self).__init__()
        # Define the GRU layer
        self.gru_layer_features = nn.GRU(input_size=input_dim, 
                                            hidden_size=hidden_dim, 
                                            num_layers=num_layers, 
                                            batch_first=True, 
                                            dropout=dropout_prob
                                        )
        self.gru_layer_cars = nn.GRU(input_size=16, 
                                        hidden_size=hidden_dim
                                        num_layers=num_layers, 
                                        batch_first=True, 
                                        dropout=dropout_prob                                    
                                    )
        self.gru_layer_pedestrians = nn.GRU(input_size=7, 
                                            hidden_size=hidden_dim
                                            num_layers=num_layers, 
                                            batch_first=True, 
                                            dropout=dropout_prob        
                                        )
        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim * 3, output_dim)
        
    def forward2(self, input_features, input_cars, input_pedestrians):
        batch_size = input_features.size(1)

        # Initialize the hidden state
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(input_features.device)
        
        # Feed the input through the GRU layer
        gru_out_features, _ = self.gru_layer(input_features, h0)
        gru_out_cars, _ = self.gru_layer(input_cars, h0)
        gru_out_pedestrians, _ = self.gru_layer(input_pedestrians, h0)

        # Combine hidden layer output
        hidden_combined = torch.cat((gru_out_features, gru_out_cars, gru_out_pedestrians), dim=2)
        
        # Feed the GRU output through the output layer
        output = self.output_layer(hidden_combined[-1, :, :])
        
        return output

    def forward(self, input_features, input_cars, input_pedestrians):
        batch_size = input_features.size(0)

        # Initialize the hidden state
        h0_features = torch.zeros(1, batch_size, self.gru_layer_features.hidden_size).to(input_features.device)
        h0_cars = torch.zeros(1, batch_size, self.gru_layer_cars.hidden_size).to(input_cars.device)
        h0_pedestrians = torch.zeros(1, batch_size, self.gru_layer_pedestrians.hidden_size).to(input_pedestrians.device)
        
        # Create masks to handle variable-length sequences and missing data
        mask_features = torch.sum(input_features != -1, dim=2).bool()
        mask_cars = torch.sum(input_cars != -1, dim=2).bool()
        mask_pedestrians = torch.sum(input_pedestrians != -1, dim=2).bool()
        
        # Pack input sequences and apply GRU layers with masks
        input_features_packed = nn.utils.rnn.pack_padded_sequence(input_features, mask_features.sum(dim=1), batch_first=True, enforce_sorted=False)
        gru_out_features_packed, _ = self.gru_layer_features(input_features_packed, h0_features)
        gru_out_features, _ = nn.utils.rnn.pad_packed_sequence(gru_out_features_packed, batch_first=True, total_length=input_features.size(1))

        input_cars_packed = nn.utils.rnn.pack_padded_sequence(input_cars, mask_cars.sum(dim=1), batch_first=True, enforce_sorted=False)
        gru_out_cars_packed, _ = self.gru_layer_cars(input_cars_packed, h0_cars)
        gru_out_cars, _ = nn.utils.rnn.pad_packed_sequence(gru_out_cars_packed, batch_first=True, total_length=input_cars.size(1))

        input_pedestrians_packed = nn.utils.rnn.pack_padded_sequence(input_pedestrians, mask_pedestrians.sum(dim=1), batch_first=True, enforce_sorted=False)
        gru_out_pedestrians_packed, _ = self.gru_layer_pedestrians(input_pedestrians_packed, h0_pedestrians)
        gru_out_pedestrians, _ = nn.utils.rnn.pad_packed_sequence(gru_out_pedestrians_packed, batch_first=True, total_length=input_pedestrians.size(1))

        # Combine hidden layer output
        hidden_combined = torch.cat((gru_out_features, gru_out_cars, gru_out_pedestrians), dim=2)
        
        # Feed the GRU output through the output layer
        output = self.output_layer(hidden_combined[-1, :, :])
        
        return output