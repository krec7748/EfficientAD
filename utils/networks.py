import torch
from torch import nn

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, batchnorm = False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, upsample_size:int, in_channels, out_channels, kernel_size = 4, stride = 1, padding = 2, batchnorm = False, dropout = True):
        super().__init__()

        self.upsample = nn.Upsample(size = upsample_size, mode = "bilinear")
        self.convrelu = ConvReLU(in_channels, out_channels, kernel_size, stride, padding, batchnorm)
        self.dropout = nn.Dropout(0.2) if dropout else None
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.convrelu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x    

class AutoEncoder(nn.Module):
    def __init__(self, out_channels = 384):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ConvReLU(3, 32, 4, stride = 2, padding = 1),
            ConvReLU(32, 32, 4, stride = 2, padding = 1),
            ConvReLU(32, 64, 4, stride = 2, padding = 1),
            ConvReLU(64, 64, 4, stride = 2, padding = 1),
            ConvReLU(64, 64, 4, stride = 2, padding = 1),
            nn.Conv2d(64, 64, 8),
        )

        # Decoder
        self.decoder = nn.Sequential(
            DecoderLayer(3, 64, 64),
            DecoderLayer(8, 64, 64), 
            DecoderLayer(15, 64, 64),
            DecoderLayer(32, 64, 64),
            DecoderLayer(63, 64, 64),
            DecoderLayer(127, 64, 64),
            DecoderLayer(56, 64, 64, 3, padding = 1, dropout = False),
            nn.Conv2d(64, out_channels, 3, padding = 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PDN(nn.Module):
    def __init__(self, model_size:str, out_channels:int = 384, padding:bool = False):
        super().__init__()

        assert model_size in ["small", "medium"]

        self.model_size = model_size
        self.out_channels = out_channels

        pad_mult = 1 if padding else 0
        channels_mult = 2 if model_size == "medium" else 1

        self.pdn = nn.Sequential(
            ConvReLU(3, 128 * channels_mult, 4, padding = 3 * pad_mult),
            nn.AvgPool2d(2, 2, padding = 1 * pad_mult),
            ConvReLU(128 * channels_mult, 256 * channels_mult, 4, padding = 3 * pad_mult),
            nn.AvgPool2d(2, 2, padding = 1 * pad_mult),
            ConvReLU(256 * channels_mult, 256 * channels_mult, 3, padding = 1 * pad_mult),
        )

        if model_size == "small":
            self.pdn.append(nn.Conv2d(256, out_channels, 4))

        elif model_size == "medium":
            self.pdn.insert(4, ConvReLU(512, 512, 1))
            self.pdn.append(ConvReLU(512, out_channels, 4))
            self.pdn.append(nn.Conv2d(out_channels, out_channels, 1))
    
    def forward(self, x):
        return self.pdn(x)

def load_weights(model:nn.Module, weights_path:str):

    weights_dict = torch.load(weights_path, map_location = "cpu")

    try:
        model.load_state_dict(weights_dict, strict = True)

    except:
        model_dict_keys = model.state_dict().keys()
        assert len(model_dict_keys) == len(weights_dict)

        new_state_dict = {}
        for k, v in zip(model_dict_keys, weights_dict.values()):
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)