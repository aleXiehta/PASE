import torch
from torch import nn
import torch.nn.init as init

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        kargs = {
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'groups': groups,
                'bias': True
                }

        self.real_conv = nn.Conv2d(**kargs)
        self.imag_conv = nn.Conv2d(**kargs)

    def forward(self, c_input):
        real, imag = c_input[..., 0], c_input[..., 1]

        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.imag_conv(real) + self.real_conv(imag)
        
        return torch.stack((real_out, imag_out), dim=-1)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ComplexConvTranspose2d, self).__init__()
        kargs = {
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'output_padding': output_padding,
                'dilation': dilation,
                'groups': groups,
                'bias': True
                }

        self.real_deconv = nn.ConvTranspose2d(**kargs)
        self.imag_deconv = nn.ConvTranspose2d(**kargs)

    def forward(self, c_input):
        real, imag = c_input[..., 0], c_input[..., 1]

        real_out = (self.real_deconv(real) + self.imag_deconv(imag))
        imag_out = self.real_deconv(imag) - self.imag_deconv(real)

        factor = 1. / (real_out.pow(2) + imag_out.pow(2))

        real_out = real_out * factor
        imag_out = imag_out * factor

        return torch.stack((real_out, imag_out), dim=-1)

#TODO Fix complex batch normalization. Using BatchNorm3d instead for now
'''
Reference:
[1] Implement Custom BatchNorm2d: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/blob/master/sync_batchnorm/batchnorm_reimpl.py
[2] Complex Covariance Function for PyTorch: https://stackoverflow.com/questions/51416825/calculate-covariance-matrix-for-complex-data-in-two-channels-no-complex-data-ty


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(ComplexBatchNorm2d, self)__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean_real', torch.zeros(num_features))
        self.register_buffer('running_mean_imag', torch.zeros(num_features))
        self.register_buffer('running_var_real', torch.zeros(num_features))
        self.register_buffer('running_var_imag', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean_real.zero_() + 1 / 2 ** 0.5
        self.running_mean_imag.zero_() + 1 / 2 ** 0.5
        self.running_var_real.zero_()
        self.running_var_imag.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, c_input):
        real, imag = c_input[..., 0], c_input[..., 1]

        centered_real = real - real.mean(dim=1)
        centered_imag = imag - imag.mean(dim=1)
        centered_squared_real = centered_real ** 2
        centered_squared_imag = centered_imag ** 2
        centered = torch.stack([centered_real, centered_imag], dim=-1)
        centered_squared = \
                torch.stack([centered_squared_real, centered_squared_imag], dim=-1])
        
        Vrr = centered_squared_real.mean(2) + self.eps
        Vii = centered_squared_imag.mean(2) + self.eps
        Vri = (centered_real * centered_imag).mean(2)

        input_bn = self.complexBN(centered_real, centered_imag, Vrr, Vii, Vri)

        if self.training:
            self.running_mean_real = \
                    (1 - self.momentum) * self.running_mean_real + self.momentum * real.mean(2)
            self.running_var = \
                    (1 - self.momentum) * self.running_var + self.momentum * var.mean(2)
            
            def normalize_inference():
                inference_centered_real = real - self.running_mean
                inference_centered_imag = imag - self.jk
        else:
            return input_bn

    def complexBN(self, centered_real, centered_imag, Vrr, Vii, Vri):
        pass

    def complexSTD(self, centered_real, centered_imag, Vrr, Vii, Vri):
        pass
    '''
