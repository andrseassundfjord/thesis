import torch
import torch.nn as nn
import torch.nn.functional as F
from components import VideoEncoder, VideoEncoderPretrained, VideoDecoder

class VideoVAE(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout):
        super(VideoVAE, self).__init__()

        self.video_input_shape = input_dims[0]
        self.hidden_shape = hidden_layers[0]
        # Encoder
        self.video_encoder = VideoEncoder(latent_dim = latent_dim, 
                                            input_shape = self.video_input_shape, 
                                            hidden_shape = self.hidden_shape,
                                            dropout = dropout,
                                            hidden_dim = hidden_layers[1]
                                        )
        # Decoder
        self.video_decoder = VideoDecoder(latent_dim = latent_dim, 
                                            input_shape = self.video_input_shape, 
                                            hidden_shape = self.hidden_shape,
                                            dropout = dropout,
                                            hidden_dim = hidden_layers[1]
                                        )

    def forward(self, video):
        # Encode video
        encoded_video = self.video_encoder(video)
        # Get KL divergence
        kl_divergence = -0.5 * torch.sum(1 + encoded_video[2] - encoded_video[1].pow(2) - encoded_video[2].exp(), dim=1).mean()
        # Decode
        decoded_video = self.video_decoder(encoded_video[0])
        return decoded_video, kl_divergence, encoded_video[0], encoded_video[1]

