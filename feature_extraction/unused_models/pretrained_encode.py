import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PretrainedEncoder(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout = 0.1):
        super(PretrainedEncoder, self).__init__()
        num_frames = input_dims[0][0]
        transformer_d_model = latent_dim * 2

        # Load a pre-trained CNN backbone (ResNet-152)
        self.backbone = models.resnet152(pretrained=True)

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add a transformer encoder layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

        # Add a new linear layer to output the latent representation
        self.fc = nn.Linear(transformer_d_model, latent_dim)

        self.decoder = VideoBERTDecoder(transformer_d_model, latent_dim, num_frames)


    def forward(self, x):
        # Reshape input from (batch_size, channels, num_frames, height, width) to (batch_size*num_frames, channels, height, width)
        batch_size = x.shape[0]
        x = x.reshape(batch_size*self.num_frames, x.shape[1], x.shape[3], x.shape[4])

        # Extract features from the input video frames using the backbone
        features = self.backbone(x)  # shape: (batch_size, num_frames, 2048)

        # Transpose the feature tensor to be compatible with the transformer input
        features = features.permute(1, 0, 2)  # shape: (num_frames, batch_size, 2048)
        print(features.size())
        # Apply the transformer encoder to the feature tensor
        encoded_features = self.transformer_encoder(features)

        # Take the last encoded frame as the representation of the input video
        encoded_video = encoded_features[-1, :, :]  # shape: (batch_size, transformer_d_model)

        # Apply the final linear layer to obtain the latent representation
        latent_representation = self.fc(encoded_video)  # shape: (batch_size, latent_dim)

        # Use the decoder to generate the predicted future frames
        predicted_frames = self.decoder(latent_representation)  # shape: (batch_size, 3, num_frames, H, W)

        return latent_representation, predicted_frames

class VideoBERTDecoder(nn.Module):
    def __init__(self, transformer_d_model, latent_dim, num_predicted_frames):
        super(VideoBERTDecoder, self).__init__()

        # Save the input parameters as class attributes
        self.transformer_d_model = transformer_d_model
        self.latent_dim = latent_dim
        self.num_predicted_frames = num_predicted_frames

        # Add a linear layer to map the latent representation to the transformer input
        self.fc1 = nn.Linear(latent_dim, transformer_d_model)
        self.fc2 = nn.Linear(latent_dim, transformer_d_model)

        # Add a transformer decoder layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_d_model, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=2)

        # Add a convolutional layer to generate the predicted future frames
        self.conv = nn.ConvTranspose2d(2048, 3, kernel_size=4, stride=2, padding=1)

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
        #decoded_frames = self.fc3(decoded_frames)
        # Transpose the decoded frames tensor to be compatible with the convolutional layer
        decoded_frames = decoded_frames.permute(1, 2, 0)  # shape: (batch_size, transformer_d_model, num_predicted_frames)

        decoded_frames = decoded_frames.reshape(decoded_frames.size(0) * decoded_frames.size(2), self.cnn_reshape, 16, 16)
        # Apply the convolutional layer to generate the predicted future frames
        predicted_frames = self.conv(decoded_frames)  # shape: (batch_size, 3, H, W)
        batch_size = encoded_video.size(0)
        predicted_frames = predicted_frames.reshape(batch_size, 3, self.num_predicted_frames, 128, 128)
        return predicted_frames
