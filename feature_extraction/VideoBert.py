import torch.nn as nn
import torch.nn.functional as F

class VideoBERT(nn.Module):
    def __init__(self, input_shape, num_classes, num_frames=64, cnn_filters=[32, 64], cnn_kernel_size=[3, 3], cnn_pool_size=[2, 2], transformer_layers=2, transformer_d_model=512, transformer_num_heads=8, transformer_dff=2048, transformer_dropout=0.1):
        super(VideoBERT, self).__init__()

        # Define 3D CNN encoder
        cnn_layers = []
        in_channels = input_shape[0]
        for i in range(len(cnn_filters)):
            cnn_layers.append(nn.Conv3d(in_channels, cnn_filters[i], kernel_size=cnn_kernel_size[i], padding=1))
            cnn_layers.append(nn.ReLU(inplace=True))
            cnn_layers.append(nn.MaxPool3d(kernel_size=cnn_pool_size[i], stride=cnn_pool_size[i]))
            in_channels = cnn_filters[i]
        self.cnn_encoder = nn.Sequential(*cnn_layers)

        # Define Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=transformer_num_heads, dim_feedforward=transformer_dff, dropout=transformer_dropout), num_layers=transformer_layers)

        # Define output layer
        self.fc = nn.Linear(transformer_d_model, num_classes)

        # Define class variables
        self.num_frames = num_frames

    def forward(self, x):
        # Reshape input from (batch_size, num_frames, height, width, channels) to (batch_size*num_frames, channels, height, width)
        batch_size = x.shape[0]
        x = x.reshape(batch_size*self.num_frames, x.shape[2], x.shape[3], x.shape[4])

        # Encode input using CNN encoder
        cnn_encoded = self.cnn_encoder(x)

        # Reshape CNN output to (batch_size, num_frames, cnn_encoded_size)
        cnn_encoded = cnn_encoded.view(batch_size, self.num_frames, -1)

        # Transpose CNN output to (num_frames, batch_size, cnn_encoded_size)
        cnn_encoded = cnn_encoded.permute(1, 0, 2)

        # Encode CNN output using Transformer encoder
        transformer_encoded = self.transformer_encoder(cnn_encoded)

        # Get last transformer encoded frame (num_frames-1) as the final representation
        transformer_encoded_last_frame = transformer_encoded[-1, :, :]

        # Apply output layer
        outputs = self.fc(transformer_encoded_last_frame)

        return outputs
