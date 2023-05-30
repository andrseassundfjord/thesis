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
    def __init__(self, latent_dim, input_shape, hidden_shape, dropout):
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
            #nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 1),
            nn.Conv3d(self.hs[0], self.hs[1], kernel_size = 4, stride = 2, padding = 1),
            #nn.BatchNorm3d(self.hs[1]),
            nn.LeakyReLU(),
            #nn.MaxPool3d(kernel_size = 3, stride = 1, padding = 2),
            nn.Conv3d(self.hs[1], self.hs[2], kernel_size = 4, stride = 2, padding = 1),
            #nn.BatchNorm3d(self.hs[2]),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size = (3, 4, 4), stride = (1, 2, 2), padding = 1),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(int(self.hs[2] * self.width * self.height * self.n_frames / (8 * 16 * 16)), self.hs[3]),
            #nn.BatchNorm1d(self.hs[3]),
            nn.LeakyReLU()
        ])
        
        self.fc = nn.Linear(self.hs[3], latent_dim)
        self.sigmoid = nn.Sigmoid()

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        #nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        #if torch.any(torch.isnan(x)):
        #    print("NaN in: Encoder input")
        for layer in self.model:
            x = layer(x)
            #if torch.any(torch.isnan(x)):
            #    print("NaN in: Encoder ", layer)
            #print(layer, x.size())
        x = x.view(x.size(0), -1)
        #latent = self.sigmoid(self.fc(x))
        latent = self.fc(x)
        return latent

class VideoEncoderPretrained(nn.Module):
    def __init__(self, latent_dim, input_shape, hidden_shape, dropout):
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

        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        if torch.any(torch.isnan(x)):
            print("NaN in: Encoder input")
        x = self.backbone(x)
        if torch.any(torch.isnan(x)):
            print("NaN in: backbone")
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        if torch.any(torch.isnan(x)):
            print("NaN in: last linear layer")
        return latent

class VideoDecoder(nn.Module):
    def __init__(self, latent_dim, input_shape, hidden_shape, dropout):
        super(VideoDecoder, self).__init__()
        self.n_frames = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.n_channel = input_shape[3]
        self.hs = hidden_shape

        self.model = nn.ModuleList([
            nn.Linear(latent_dim, self.hs[3]),
            nn.BatchNorm1d(self.hs[3]),
            nn.ReLU(),
            nn.Linear(self.hs[3], int(self.hs[2] * self.width * self.height * self.n_frames / (8 ** 3))),
            #nn.BatchNorm1d(int(self.hs[2] * self.width * self.height * self.n_frames / (8 ** 3))),
            nn.ReLU(),
            nn.Dropout(dropout),
            Reshape((self.hs[2], int(self.n_frames / 8), int(self.width / 8), int(self.height / 8))),
            nn.ConvTranspose3d(self.hs[2], self.hs[1], kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm3d(self.hs[1]),
            nn.ReLU(),
            nn.ConvTranspose3d(self.hs[1], self.hs[0], kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm3d(self.hs[0]),
            nn.ReLU(),
            nn.ConvTranspose3d(self.hs[0], self.n_channel, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm3d(self.n_channel),
            nn.Sigmoid()
        ])

        # Weight init
        for m in self.model:
            if isinstance(m, (nn.ConvTranspose3d, nn.Linear)):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, x):
        #if torch.any(torch.isnan(x)):
        #    print("NaN in: Decoder input")
        for layer in self.model:
            x = layer(x)
        #    if torch.any(torch.isnan(x)):
        #        print("NaN in: Decoder ", layer)
            #print(layer, x.size())
        return x

class VideoAutoencoder(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_layers, dropout):
        super(VideoAutoencoder, self).__init__()

        self.video_input_shape = input_dims[0]
        self.hidden_shape = hidden_layers[0]
        # Encoder
        self.video_encoder = VideoEncoder(latent_dim = latent_dim, 
                                            input_shape = self.video_input_shape, 
                                            hidden_shape = self.hidden_shape,
                                            dropout = dropout
                                        )
            
        # Decoder
        self.video_decoder = VideoDecoder(latent_dim = latent_dim, 
                                            input_shape = self.video_input_shape, 
                                            hidden_shape = self.hidden_shape,
                                            dropout = dropout
                                        )

    def forward(self, video):
        # Encode video
        
        encoded_video = self.video_encoder(video)
        # Decode
        decoded_video = self.video_decoder(encoded_video)
        return decoded_video, encoded_video

