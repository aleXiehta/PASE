import torch
import torch.nn as nn
from .modules import ConvBlock, DeconvBlock

class DCUnet10(nn.Module):
    def __init__(self, n_channel):
        super(DCUnet10, self).__init__()
        self.conv1 = ConvBlock(n_channel, 32, (7, 5), (2, 2))
        self.conv2 = ConvBlock(32, 64, (7, 5), (2, 2))
        self.conv3 = ConvBlock(64, 64, (5, 3), (2, 2))
        self.conv4 = ConvBlock(64, 64, (5, 3), (2, 2))
        self.conv5 = ConvBlock(64, 64, (5, 3), (2, 1))
        self.conv6 = ConvBlock(64, 64, (5, 1), (1, 1))

        self.deconv1 = DeconvBlock(64 * 2, 64, (5, 3), (2, 1))
        self.deconv2 = DeconvBlock(64 * 2, 64, (5, 3), (2, 2))
        self.deconv3 = DeconvBlock(64 * 2, 64, (5, 3), (2, 2))
        self.deconv4 = DeconvBlock(64 * 2, 32, (7, 5), (2, 2))
        self.deconv5 = DeconvBlock(32 * 2, n_channel, (7, 5), (2, 2))

    def forward(self, c_input):
        c1 = self.conv1(c_input)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        x = self.deconv1(c5, c4)
        x = self.deconv2(x, c3)
        x = self.deconv3(x, c2)
        x = self.deconv4(x, c1)
        
