import torch
from model.modules import ConvBlock

frame_in = torch.randn(32, 1, 257, 16, 2)
net = ConvBlock(1, 10, (7, 1), (2, 2))

frame_out = net(frame_in)
