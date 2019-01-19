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

        # trainable
        self.gamma_rr = nn.Parameter(torch.ones(num_features) / (SQRT_2), requires_grad=True)
        self.gamma_ii = nn.Parameter(torch.ones(num_features) / (SQRT_2), requires_grad=True)
        self.gamma_ri = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.beta_real = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.beta_imag = nn.Parameter(torch.zeros(num_features), requires_grad=True)

        # non-tranable
        self.running_mean_real = torch.ones(num_features) / (SQRT_2)
        self.running_mean_imag = torch.ones(num_features) / (SQRT_2)
        self.running_Vrr = torch.ones(num_features) / (SQRT_2)
        self.running_Vii = torch.ones(num_features) / (SQRT_2)
        self.running_Vri = torch.zeros(num_features)

    def reset_running_stats(self):
        self.running_mean_real.zero_() + 1 / SQRT_2
        self.running_mean_imag.zero_() + 1 / SQRT_2
        self.running_Vrr.zero_() + 1 / SQRT_2
        self.running_Vii.zero_() + 1 / SQRT_2
        self.running_Vri.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        self.gamma_rr.zero_() + 1 / SQRT_2
        self.gamma_ii.zero_() + 1 / SQRT_2
        self.gamma_ri.zero_()
        self.beta_real.zero_()
        self.beta_imag.zero_()

    def update_running_parameters(self, real_mean, imag_mean, Vrr, Vri, Vii, ch):
        self.running_mean_real[ch] = self.momentum * self.running_mean_real[ch] + (1. - self.momentum) * real_mean
        self.running_mean_imag[ch] = self.momentum * self.running_mean_imag[ch] + (1. - self.momentum) * imag_mean
        self.running_Vrr[ch] = self.momentum * self.running_Vrr[ch] + (1. - self.momentum) * Vrr[ch]
        self.running_Vri[ch] = self.momentum * self.running_Vri[ch] + (1. - self.momentum) * Vri[ch]
        self.running_Vii[ch] = self.momentum * self.running_Vii[ch] + (1. - self.momentum) * Vii[ch]

    def forward(self, c_input):
        Vrr = torch.ones(self.num_features) / (SQRT_2)
        Vii = torch.ones(self.num_features) / (SQRT_2)
        Vri = torch.zeros(self.num_features)

        real_in, imag_in = c_input[..., 0], c_input[..., 1]

        outputs = []

        for ch in range(self.num_features):
            real = real_in[:, ch, :, :]
            imag = imag_in[:, ch, :, :]

            self.input_shape = real.shape

            real, imag = real.contiguous().view(-1,), imag.contiguous().view(-1,)

            print(real.shape)

            if self.training:
                real_mean, imag_mean = real.mean(), imag.mean()

                centered_real = real - real_mean
                centered_imag = imag - imag_mean

                centered_real.unsqueeze_(0)
                centered_imag.unsqueeze_(0)

                Vrr[ch] = torch.mm(centered_real.t(), centered_real).mean() + self.eps
                Vri[ch] = torch.mm(centered_real.t(), centered_imag).mean() + self.eps
                Vii[ch] = torch.mm(centered_imag.t(), centered_imag).mean() + self.eps

                self.update_running_parameters(real_mean, imag_mean, Vrr, Vri, Vii, ch)

                centered_real.squeeze_(0)
                centered_imag.squeeze_(0)

                outputs.append(self.ComplexBN(centered_real, centered_imag, Vrr, Vri ,Vii, ch))

            else:
                centered_real = real - self.running_mean_real
                centered_imag = real - self.running_mean_imag

                outputs.append(self.ComplexBN(centered_real, centered_imag, self.running_Vrr, self.running_Vri, self.running_Vii, ch))

        return torch.stack(outputs, dim=0).permute(1, 0, 2, 3, 4)

    def ComplexBN(self, centered_real, centered_imag, Vrr, Vii, Vri, ch):
        # Compute square root and inverse of V
        tau = Vrr[ch] + Vii[ch] # trace, >= 0 because semi positive definite
        delta = (Vrr[ch] * Vii[ch]) - (Vri[ch] ** 2)

        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)

        inv_V = torch.Tensor([[Vii[ch] + s, -Vri[ch]], [-Vri[ch], Vrr[ch] + s]]) / (s * t)
        x_hat = torch.mm(inv_V, torch.stack([centered_real, centered_imag], dim=0)) # shape=(2, b*h*w)

        res_real = self.gamma_rr[ch] * x_hat[0] + self.gamma_ri[ch] * x_hat[1] + self.beta_real[ch]
        res_imag = self.gamma_ri[ch] * x_hat[0] + self.gamma_ii[ch] * x_hat[1] + self.beta_imag[ch]

        real, imag = res_real.view(self.input_shape), res_imag.view(self.input_shape)

        return torch.stack((real, imag), dim=-1)