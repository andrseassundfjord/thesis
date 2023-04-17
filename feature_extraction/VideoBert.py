import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class VideoBERT(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout = 0.1):
        super(VideoBERT, self).__init__()
        kernel_size = (1, 4, 4)
        stride = (1, 2, 2)
        padding = (0, 1, 1)
        transformer_layers=2
        output_size = input_dims[0][1]
        for _ in range(3):
            output_size = int((output_size - kernel_size[1] + 2*padding[1])/stride[1]) + 1
        self.transformer_d_model = 1024
        transformer_num_heads = 8
        transformer_dff = 2048
        transformer_dropout = dropout
        cnn_filters = hidden_layers[0][1:]
        cnn_layers = []
        in_channels = input_dims[0][3]
        # Define 3D CNN encoder
        for i in range(len(cnn_filters)):
            cnn_layers.append(nn.Conv3d(in_channels, cnn_filters[i], kernel_size = kernel_size, stride = stride, padding = padding))
            cnn_layers.append(nn.LeakyReLU(inplace=True))
            cnn_layers.append(nn.BatchNorm3d(cnn_filters[i]))
            in_channels = cnn_filters[i]
        self.cnn_encoder = nn.Sequential(*cnn_layers)

        self.fc_middle = nn.Linear(in_channels * output_size ** 2, self.transformer_d_model)
        # Define Transformer encoder
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

        for m in self.cnn_encoder:
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                #init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        # Add a new linear layer to output the latent representation
        self.fc = nn.Linear(self.transformer_d_model, latent_dim)

        self.decoder = VideoBERTDecoder(self.transformer_d_model, latent_dim, input_dims[0][0], cnn_filters=cnn_filters,
                                        kernel_size = kernel_size, stride = stride, padding = padding, input_size = output_size)
        # Define class variables
        self.num_frames = input_dims[0][0]

    def forward(self, x):
        # Reshape input from (batch_size, channels, num_frames, height, width) to (batch_size*num_frames, channels, height, width)
        batch_size = x.shape[0]
        num_frames = x.shape[2]
        #x = x.reshape(batch_size*num_frames, x.shape[1], x.shape[3], x.shape[4])

        # Encode input using CNN encoder
        cnn_encoded = self.cnn_encoder(x)
        # Reshape CNN output to (batch_size, num_frames, cnn_encoded_size)
        cnn_encoded = cnn_encoded.view(batch_size, num_frames, -1)
        # Transpose the feature tensor to be compatible with the transformer input
        cnn_encoded = cnn_encoded.permute(1, 0, 2)  # shape: (num_frames, batch_size, transformer_d_model)
        cnn_encoded = self.fc_middle(cnn_encoded)
        # Encode CNN output using Transformer encoder
        encoded_features = self.transformer_encoder(cnn_encoded)
        # Take the last encoded frame as the representation of the input video
        encoded_video = encoded_features[-1, :, :]  # shape: (batch_size, transformer_d_model)
        # Apply the final linear layer to obtain the latent representation
        latent_representation = self.fc(encoded_video)  # shape: (batch_size, latent_dim)
        # Use the decoder to generate the predicted future frames
        predicted_frames = self.decoder(latent_representation)  # shape: (batch_size, 3, num_frames, H, W)
        return latent_representation, predicted_frames

class VideoBERTDecoder(nn.Module):
    def __init__(self, transformer_d_model, latent_dim, num_predicted_frames, cnn_filters, kernel_size, stride, padding, input_size):
        super(VideoBERTDecoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Save the input parameters as class attributes
        self.transformer_d_model = transformer_d_model
        self.latent_dim = latent_dim
        self.num_predicted_frames = num_predicted_frames
        self.input_size = input_size
        transformer_layers = 2
        transformer_num_heads = 8
        transformer_dff = 2048

        self.cnn_reshape = cnn_filters[-1]

        # Add a linear layer to map the latent representation to the transformer input
        self.fc1 = nn.Linear(latent_dim, transformer_d_model)
        self.fc2 = nn.Linear(latent_dim, transformer_d_model)
        self.fc3 = nn.Linear(transformer_d_model, cnn_filters[-1] * input_size ** 2)

        # Add a transformer decoder layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_d_model, 
                                                                    nhead=transformer_num_heads,
                                                                    dim_feedforward=transformer_dff)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=transformer_layers)

        # Add a convolutional layer to generate the predicted future frames
        cnn_layers = []
        in_channels = cnn_filters[-1]
        cnn_filters = cnn_filters[:-1]
        cnn_filters.reverse()
        cnn_filters.append(3)
        for i in range(len(cnn_filters)):
            cnn_layers.append(nn.ConvTranspose3d(in_channels, cnn_filters[i], kernel_size = kernel_size, stride = stride, padding = padding))
            cnn_layers.append(nn.LeakyReLU(inplace=True))
            cnn_layers.append(nn.BatchNorm3d(cnn_filters[i]))
            in_channels = cnn_filters[i]
        self.conv = nn.Sequential(*cnn_layers)
        for m in self.conv:
            if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, encoded_video):
        # Apply the first linear layer to obtain the transformer input
        transformer_input = self.fc1(encoded_video)  # shape: (batch_size, transformer_d_model)
        # Expand the transformer input to have the same length as the predicted frames
        transformer_input = transformer_input.unsqueeze(1).expand(-1, self.num_predicted_frames, -1)  # shape: (batch_size, num_predicted_frames, transformer_d_model)
        transformer_input = transformer_input.permute(1, 0, 2)

        # Generate a mask tensor to prevent the decoder from attending to future frames
        mask = torch.ones(self.num_predicted_frames, self.num_predicted_frames)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0)).to(self.device)
        # Permute input
        memory = self.fc2(encoded_video)
        memory = memory.unsqueeze(0).expand(self.num_predicted_frames, -1, -1)

        # Apply the transformer decoder to generate the predicted future frames
        decoded_frames = self.transformer_decoder(transformer_input, memory, tgt_mask=mask)  # shape: (num_predicted_frames, batch_size, transformer_d_model)
        decoded_frames = self.fc3(decoded_frames)
        # Transpose the decoded frames tensor to be compatible with the convolutional layer
        decoded_frames = decoded_frames.permute(1, 2, 0)  # shape: (batch_size, transformer_d_model, num_predicted_frames)

        decoded_frames = decoded_frames.reshape(decoded_frames.size(0), self.cnn_reshape, decoded_frames.size(2), self.input_size, self.input_size)
        # Apply the convolutional layer to generate the predicted future frames
        predicted_frames = self.conv(decoded_frames)  # shape: (batch_size, 3, num_frames, H, W)

        return predicted_frames