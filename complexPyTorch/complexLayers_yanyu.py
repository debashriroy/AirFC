#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:02 2019

@author: Sebastien M. Popoff


Based on https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d
from complexPyTorch.complexFunctions_yanyu import complex_relu, complex_max_pool2d
from complexPyTorch.complexFunctions_yanyu import complex_dropout, complex_dropout2d
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

constraints0 = torch.load('weights/real_img_num0.pt')
constraints1 = torch.load('weights/real_img_num1.pt')
constraints2 = torch.load('weights/real_img_num2.pt')
constraints = (constraints0, constraints1, constraints2)


def append_triple(real, img, constraints):
    split_real = real.split(1, dim=2)
    split_img = img.split(1, dim=2)

    list_real = []
    list_img = []
    for i in range(3):
        a, b = append_min_distance(split_real[i], split_img[i], constraints[i])
        list_real.append(a)
        list_img.append(b)
    return torch.cat(list_real, dim=2), torch.cat(list_img, dim=2)


def append_min_distance(real, img, constraints):
    # print("Printing : ", real, img)
    if real is not None:
        w1 = real.detach()
    size1 = w1.size()
    flat1 = w1.reshape(-1)

    if img is not None:
        w2 = img.detach()
    size2 = w2.size()
    flat2 = w2.reshape(-1)

    cons = constraints.to(real.device)

    distances_real = flat1 - cons[0].reshape(-1, 1)
    distances_img = flat2 - cons[1].reshape(-1, 1)
    h_distance = torch.abs(distances_real) + torch.abs(distances_img)

    idx = torch.argmin(h_distance, dim=0)

    sub_real = torch.gather(distances_real, 0, idx.reshape(1, -1))
    sub_img = torch.gather(distances_img, 0, idx.reshape(1, -1))

    return sub_real.reshape(size1), sub_img.reshape(size2)


class Conv2dNoise(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, var=None, pruning=False, pruningRate=0.):
        super(Conv2dNoise, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.var = var
        self.pruning = pruning

    # @weak_script_method
    def forward(self, input, adjustment_w=None, adjustment_b=None):
        if adjustment_w is not None:
            w = self.weight - adjustment_w.float()
        else:
            w = self.weight

        if adjustment_b is not None:
            b = self.bias - adjustment_b.float().to(self.bias.device)
        else:
            b = self.bias

        if self.var is not None:
            n = torch.randn(w.size()) * torch.sqrt(self.var)
            # if torch.gt(torch.sum(torch.gt(n, torch.tensor(0.0553))), 0):
            #     print("It Happeneeeeed")
            #     print(n)
            # while torch.gt(torch.sum(torch.gt(n, torch.tensor(0.0553))), 0):
            #     n = torch.randn(w.size()) * torch.sqrt(self.var)
            # w = w + (torch.randn(w.size()) * torch.sqrt(self.var)).to(w.device)
            if self.pruning:
                mask = (~torch.eq(w, torch.tensor(0))).long()
                n = n.to(w.device) * mask
            w = w + n.to(w.device)
        output = F.conv2d(input, w, b, self.stride, self.padding, self.dilation, self.groups)
        return output


def quantization(input, bit=6):
    no_grad = input.detach()
    sign = torch.sign(no_grad)
    scale = torch.max(torch.abs(no_grad))
    unified = torch.abs(no_grad) / scale

    unit = float(2 ** bit)
    quantized = torch.round(unified * unit) / unit
    target = quantized * sign
    residual = target - no_grad
    return input + residual


class LinearNoise(nn.Linear):
    """docstring for QuanConv"""

    def __init__(self, in_features, out_features, varWeights=None, varNoise=None, pruning=False,
                 pruningRate=1, bias = False):
        super(LinearNoise, self).__init__(in_features, out_features, bias=False)
        self.varWeights = varWeights
        self.varNoise = varNoise
        # self.numRIS = numRIS
        self.pruning = pruning
        self.pruningRate = pruningRate

    # @weak_script_method
    def forward(self, input, adjustment_w=None, adjustment_b=None):
        if adjustment_w is not None:
            # print(adjustment)
            w = self.weight - adjustment_w.float().to(self.weight.device)
            # w = quantization(self.weight)
        else:
            w = self.weight

        if adjustment_b is not None:
            # print(adjustment)
            b = self.bias - adjustment_b.float().to(self.bias.device)
            # w = quantization(self.weight)
        else:
            b = self.bias

        # print(w)
        # inputSize = input.shape[1].detach().cpu().numpy()
        inputSize = np.array(input.shape)[1]
        # if self.varWeights
        if self.varWeights is not None:

            n = torch.randn(w.size()) * torch.sqrt(self.varWeights)

            if self.pruning:
                mask = (~torch.eq(w, torch.tensor(0))).long()
                n = n.to(w.device) * mask
            # w = w + n.to(w.device)
            w = w + n.to(w.device)


            # if (inputSize * self.pruningRate) > self.numRIS:
            #     for i in range(
            #             int(np.ceil((np.log(inputSize) * self.pruningRate) / np.log(self.numRIS))) - 1):
            #         w = w * (
            #                 1 + torch.randn(w.size()) * torch.sqrt(self.varWeights)).to(w.device)
        #
        # print(torch.max(w), torch.min(w))
        output = F.linear(input, w) # removed bias

        if self.varNoise is not None:
            z = torch.randn(1, output.size()[1]) * torch.sqrt(self.varNoise)
            return output + z.to(output.device)

        return output


class ComplexSequential(Sequential):
    def forward(self, input_r, input_t):
        for module in self._modules.values():
            input_r, input_t = module(input_r, input_t)
        return input_r, input_t


class ComplexDropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input_r, input_i):
        return complex_dropout(input_r, input_i, self.p, self.inplace)


class ComplexDropout2d(Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout2d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input_r, input_i):
        return complex_dropout2d(input_r, input_i, self.p, self.inplace)


class ComplexMaxPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input_r, input_i):
        return complex_max_pool2d(input_r, input_i, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, ceil_mode=self.ceil_mode,
                                  return_indices=self.return_indices)


class ComplexReLU(Module):

    def forward(self, input_r, input_i):
        return complex_relu(input_r, input_i)


class ComplexConvTranspose2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input_r, input_i):
        return self.conv_tran_r(input_r) - self.conv_tran_i(input_i), \
               self.conv_tran_r(input_i) + self.conv_tran_i(input_r)


class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        #        assert(input_r.size() == input_i.size())
        return self.conv_r(input_r) - self.conv_i(input_i), \
               self.conv_r(input_i) + self.conv_i(input_r)


class ComplexConv2dNoise(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, realNoise=0.000169, imagNoise=0.0000109,
                 padding=0, dilation=1, groups=1, bias=False, quant=False, pruning=False, pruningRate=0., setSize=None):
        super(ComplexConv2dNoise, self).__init__()

        # self.realv = torch.tensor(0.0001866, requires_grad=False)
        # self.imgv = torch.tensor(0.00001728, requires_grad=False)
        # self.realv = torch.tensor(0.000169, requires_grad=False)
        # self.imgv = torch.tensor(0.0000109, requires_grad=False)
        # self.realv = torch.tensor(realNoise, requires_grad=False)
        # self.imgv = torch.tensor(imagNoise, requires_grad=False)
        self.realv = None
        self.imgv = None
        self.quantization = quant
        if self.quantization:
            if setSize is None:
                # self.cons = constraints0
                self.cons = torch.randn(2, 256)
            else:
                # self.cons = modify_constraints(torch.load('weights/processed' + str(setSize) + '.pt'), 256)
                self.cons = torch.load('weights/processed' + str(setSize) + '.pt')

        self.conv_r = Conv2dNoise(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                  self.realv, pruning=pruning, pruningRate=pruningRate)
        self.conv_i = Conv2dNoise(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                  self.imgv, pruning=pruning, pruningRate=pruningRate)

    def forward(self, input_r, input_i):
        if self.quantization:
            sub_real, sub_img = append_min_distance(self.conv_r.weight, self.conv_i.weight, self.cons)
            sub_real_b, sub_img_b = append_min_distance(self.conv_r.bias, self.conv_i.bias, self.cons)
            return self.conv_r(input_r, sub_real, sub_real_b) - self.conv_i(input_i, sub_img, sub_img_b), \
                   self.conv_r(input_i, sub_real, sub_real_b) + self.conv_i(input_r, sub_img, sub_img_b)
        else:
            return self.conv_r(input_r) - self.conv_i(input_i), \
                   self.conv_r(input_i) + self.conv_i(input_r)


class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        return self.fc_r(input_r) - self.fc_i(input_i), \
               self.fc_r(input_i) + self.fc_i(input_r)


class ComplexLinearNoise(Module):

    def __init__(self, in_features, out_features, realWeightVar=0.000169, imgWeightVar=0.0000109, realNoiseVar=0.000169,
                 imgNoiseVar=0.0000109, quant=False, pruning=False, pruningRate=1, setSize=None, bias=False):
        super(ComplexLinearNoise, self).__init__()

        self.quantization = quant

        # Weights Variance
        if realWeightVar is None:
            self.realWVar = realWeightVar
        else:
            self.realWVar = torch.tensor(realWeightVar, requires_grad=False)
        if imgWeightVar is None:
            self.imgWVar = imgWeightVar
        else:
            self.imgWVar = torch.tensor(imgWeightVar, requires_grad=False)

        if realNoiseVar is None:
            self.realNoiseVar = realNoiseVar
        else:
            self.realNoiseVar = torch.tensor(realNoiseVar, requires_grad=False)
        if imgNoiseVar is None:
            self.imgNoiseVar = imgNoiseVar
        else:
            self.imgNoiseVar = torch.tensor(imgNoiseVar, requires_grad=False)

        self.fc_r = LinearNoise(in_features, out_features, varWeights=self.realWVar, varNoise=self.realNoiseVar,
                                pruning=pruning, pruningRate=pruningRate, bias=bias)
        self.fc_i = LinearNoise(in_features, out_features, varWeights=self.imgWVar, varNoise=self.imgNoiseVar,
                                 pruning=pruning, pruningRate=pruningRate, bias=bias)

        # self.cons = constraints
        # self.cons = torch.randn(2, 512)

        # self.cons = torch.load('weights/processed' + str(setSize) + '.pt')[:, 0:setSize]
        # self.cons = torch.load('weights/processed' + str(2) + '.pt')[:, 0:2]
        if self.quantization:
            if setSize is None:
                # self.cons = constraints0
                self.cons = torch.randn(2, 256)
            else:
                # self.cons = modify_constraints(torch.load('weights/processed' + str(setSize) + '.pt'), 256)
                self.cons = torch.load('weights/processed' + str(setSize) + '.pt')

    def forward(self, input_r, input_i):
        if self.quantization:
            sub_real, sub_img = append_min_distance(self.fc_r.weight, self.fc_i.weight, self.cons)
            # sub_real_b, sub_img_b = append_min_distance(self.fc_r.bias, self.fc_i.bias, self.cons)
            return self.fc_r(input_r, sub_real) - self.fc_i(input_i, sub_img), \
                   self.fc_r(input_i, sub_real) + self.fc_i(input_r, sub_img)
        else:
            return self.fc_r(input_r) - self.fc_i(input_i), \
                   self.fc_r(input_i) + self.fc_i(input_r)


class NaiveComplexBatchNorm1d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)


class NaiveComplexBatchNorm2d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)


class NaiveComplexBatchNorm1d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)


class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert (input_r.size() == input_i.size())
        assert (len(input_r.shape) == 4)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])

            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r - mean_r[None, :, None, None]
            input_i = input_i - mean_i[None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])

            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

            input_r = input_r - mean[None, :, 0, None, None]
            input_i = input_i - mean[None, :, 1, None, None]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * input_i, \
                           Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r

        if self.affine:
            input_r, input_i = self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 0, None, None], \
                               self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 1, None, None]

        return input_r, input_i


class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert (input_r.size() == input_i.size())
        assert (len(input_r.shape) == 2)
        # self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean(dim=0)
            mean_i = input_i.mean(dim=0)
            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

            # zero mean values
            input_r = input_r - mean_r[None, :]
            input_i = input_i - mean_i[None, :]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = input_r.var(dim=0, unbiased=False) + self.eps
            Cii = input_i.var(dim=0, unbiased=False) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=0)

            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]
            # zero mean values
            input_r = input_r - mean[None, :, 0]
            input_i = input_i - mean[None, :, 1]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :] * input_r + Rri[None, :] * input_i, \
                           Rii[None, :] * input_i + Rri[None, :] * input_r

        if self.affine:
            input_r, input_i = self.weight[None, :, 0] * input_r + self.weight[None, :, 2] * input_i + \
                               self.bias[None, :, 0], \
                               self.weight[None, :, 2] * input_r + self.weight[None, :, 1] * input_i + \
                               self.bias[None, :, 1]

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input_r, input_i


def modify_constraints(cons, subset_size):
    input = torch.ones((cons.size(1)))
    idx = torch.multinomial(input, subset_size)
    new_cons = torch.index_select(cons, 1, idx)
    # new_cons = []
    # for const in cons:
    #     input = torch.ones(const.size(1))
    #     idx = torch.multinomial(input, subset_size)
    #     new_cons.append(torch.index_select(const, 1, idx))
    # print('length of current subset: ', new_cons[0].size(1))
    return new_cons


# out = modify_constraints(torch.randn(2, 512), 256)
# print(out.size())
