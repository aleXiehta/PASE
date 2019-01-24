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
[2] Running average: https://www.zhihu.com/question/55621104
'''

SQRT_2 = 2 ** 0.5

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, training=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = training

        # trainable and non-trainable
        self.reset_parameters()

    def reset_running_stats(self):
        self.register_buffer('running_mean_real', torch.ones(self.num_features) / SQRT_2)
        self.register_buffer('running_mean_imag', torch.ones(self.num_features) / SQRT_2)
        self.register_buffer('running_Vrr', torch.ones(self.num_features) / SQRT_2)
        self.register_buffer('running_Vii', torch.ones(self.num_features) / SQRT_2)
        self.register_buffer('running_Vri', torch.zeros(self.num_features))

    def reset_parameters(self):
        self.reset_running_stats()

        self.gamma_rr = nn.Parameter(torch.ones(self.num_features) / (SQRT_2), requires_grad=True)
        self.gamma_ii = nn.Parameter(torch.ones(self.num_features) / (SQRT_2), requires_grad=True)
        self.gamma_ri = nn.Parameter(torch.zeros(self.num_features), requires_grad=True)
        self.beta_real = nn.Parameter(torch.zeros(self.num_features), requires_grad=True)
        self.beta_imag = nn.Parameter(torch.zeros(self.num_features), requires_grad=True)

    def update_running_parameters(self, real_mean, imag_mean, Vrr, Vri, Vii):
        self.running_mean_real = self.momentum * self.running_mean_real + (1. - self.momentum) * real_mean
        self.running_mean_imag = self.momentum * self.running_mean_imag + (1. - self.momentum) * imag_mean
        self.running_Vrr = self.momentum * self.running_Vrr + (1. - self.momentum) * Vrr
        self.running_Vri = self.momentum * self.running_Vri + (1. - self.momentum) * Vri
        self.running_Vii = self.momentum * self.running_Vii + (1. - self.momentum) * Vii

    def forward(self, c_input):
        Vrr = torch.zeros(self.num_features)
        Vii = torch.zeros(self.num_features)
        Vri = torch.zeros(self.num_features)

        real_in, imag_in = c_input[..., 0], c_input[..., 1]

        N, C, H, W = real_in.shape

        real_mean = real_in.mean((0, 2, 3)) # mean for each channel
        imag_mean = imag_in.mean((0, 2, 3))

        centered_real = real_in - real_mean.view(1, C, 1, 1) # expand dim to fit the shape of inputs
        centered_imag = real_in - imag_mean.view(1, C, 1, 1)

        # frac = 1 / (H * W - 1)

        if self.training:
            Vrr = (centered_real ** 2).mean((0, 2, 3)) + self.eps
            Vri = (centered_real.mul(centered_imag)).mean((0, 2, 3)) #+ self.eps
            Vii = (centered_imag ** 2).mean((0, 2, 3)) + self.eps

            self.update_running_parameters(real_mean, imag_mean, Vrr, Vri, Vii)

            ret = self.complexBN(centered_real, centered_imag, Vrr, Vri, Vii)

        else:
            centered_real = real - self.running_mean_real.view(1, C, 1, 1)
            centered_imag = real - self.running_mean_imag.view(1, C, 1, 1)

            ret = self.complexBN(centered_real, centered_imag, self.running_Vrr, self.running_Vri, self.running_Vii)

        return ret

    def complexBN(self, centered_real, centered_imag, Vrr, Vri, Vii):
        real_std, imag_std = self.complexSTD(centered_real, centered_imag, Vrr, Vri ,Vii)

        N, C, H, W = real_std.shape
        
        res_real = self.gamma_rr.view(1, C, 1, 1) * real_std + \
                   self.gamma_ri.view(1, C, 1, 1) * imag_std + self.beta_real.view(1, C, 1, 1)
        res_imag = self.gamma_ri.view(1, C, 1, 1) * real_std + \
                   self.gamma_ii.view(1, C, 1, 1) * imag_std + self.beta_imag.view(1, C, 1, 1)

        return torch.stack((res_real, res_imag), dim=-1)

    def complexSTD(self, centered_real, centered_imag, Vrr, Vri, Vii):
        N, C, H, W = centered_real.shape

        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)

        s = delta ** 0.5
        t = (tau + 2 * s) ** 0.5

        inverse_st = 1. / (s * t)

        print('Vrr', Vrr)
        print('Vri', Vri)
        print('Vii', Vii)

        print('delta', delta)
        print('tau', tau)
        print('s', s)
        print('t', t)
        print('inverse_st', inverse_st)

        Wrr = ((Vii + s) * inverse_st).view(1, C, 1, 1)
        Wii = ((Vrr + s) * inverse_st).view(1, C, 1, 1)
        Wri = (-Vri * inverse_st).view(1, C, 1, 1)

        assert not any(delta.view(-1,) < 0)

        output_real = Wrr * centered_real + Wri * centered_imag
        output_imag = Wri * centered_real + Wii * centered_imag

        return output_real, output_imag