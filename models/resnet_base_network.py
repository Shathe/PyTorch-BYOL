import torchvision.models as models
import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from models.mlp_head import MLPHead, MLPHead_pl, MLPHead_BNM, MLPHead_DINO
import pytorch_lightning as pl
from models import resnet_bnm

class ResNet_BN_mom(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet_BN_mom, self).__init__()
        if kwargs['name'] == 'resnet18':
            print('resnet18')
            resnet = resnet_bnm.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            print('resnet 50')
            resnet = resnet_bnm.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead_BNM(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])

        return self.projetion(h)

class ResNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__()
        if kwargs['name'] == 'resnet18':
            print('resnet18')
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            print('resnet 50')
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead_DINO(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])

        return self.projetion(h)


class ResNet_pl(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs['name'] == 'resnet18':
            print('resnet18')
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            print('resnet 50')
            resnet = models.resnet50(pretrained=False)
        self.prediction_bool = kwargs['prediction']
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead_pl(in_channels=resnet.fc.in_features, **kwargs['projection_head'])
        if self.prediction_bool:
            self.prediction = MLPHead_pl(in_channels=self.projetion.net[-1].out_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        h = self.projetion(h)
        if self.prediction_bool:
            h = self.prediction(h)

        return h

    def forward_encoder(self, x):
        h = self.encoder(x)

        return h


class MLPmixer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(MLPmixer, self).__init__()


        self.encoder = MLPMixer(in_channels=3, image_size=96, patch_size=16, num_classes=1000,
                 dim=512, depth=8, token_dim=256, channel_dim=2048)


        self.projetion = MLPHead(in_channels=512, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        return self.projetion(h)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return x

