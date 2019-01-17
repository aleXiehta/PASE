import torch
import torch.nn as nn
import torch.nn.functional as F
from .complex_layer import ComplexConv2d, ComplexConvTranspose2d


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                ComplexConv2d(in_ch, out_ch, 3, 1, 1)
        )
        # self.conv = ComplexConv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, c_input):
        c_output = self.conv(c_input)

        return c_output
