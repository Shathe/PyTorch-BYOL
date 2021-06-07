import os
from shutil import copyfile
from models.resnet_base_network import MLPMixer
import torch
import numpy as np

img = torch.ones([1, 3, 96, 96])

model = MLPMixer(in_channels=3, image_size=96, patch_size=16, num_classes=1000,
                 dim=1024, depth=8, token_dim=512, channel_dim=2046)



parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

out_img = model(img)

print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]

