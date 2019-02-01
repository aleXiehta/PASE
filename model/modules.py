import torch
import torch.nn as nn
import torch.nn.functional as F
from .complex_layer import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d 


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(in_ch, out_ch, kernel_size, stride),
            ComplexBatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self, c_input):
        c_output = self.conv(c_input)

        return c_output

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(DeconvBlock, self).__init__()
        self.deconv = ComplexuConvTranspose2d(in_ch, out_ch, kernel_size, stride)
        self.conv = ComplexConv2d(

    def foward(self, c_input, c_residual):
        c_output = self.deconv(c_input)
        c_output = torch.cat([c_output, c_residual], dim=1)

        return c_output
