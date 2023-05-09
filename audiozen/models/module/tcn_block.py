"""
contains building blocks used for the DNN-based estimators
mostly based on https://github.com/naplab/Conv-TasNet
"""

import numpy as np
import torch
from torch import nn

from audiozen.constant import EPSILON


class cLN(nn.Module):
    """cumulative layer normalization"""

    def __init__(self, dimension, eps=EPSILON, trainable=True):
        super().__init__()

        self.eps = eps
        self.gain = nn.Parameter(torch.ones((1, dimension, 1), requires_grad=trainable))
        self.bias = nn.Parameter(
            torch.zeros((1, dimension, 1), requires_grad=trainable)
        )

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )


class DepthConv1d(nn.Module):
    """depthwise separable convolution"""

    def __init__(
        self,
        input_channel,
        hidden_channel,
        kernel,
        padding,
        dilation=1,
        skip=True,
        causal=False,
    ):
        super().__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=EPSILON)
            self.reg2 = cLN(hidden_channel, eps=EPSILON)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=EPSILON)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=EPSILON)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(
                self.nonlinearity2(self.dconv1d(output)[:, :, : -self.padding])
            )
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class TCNEstimator(nn.Module):
    """
    small modification of TCN for spectrum-based parameter estimation
    based on TCN implementation in https://github.com/naplab/Conv-TasNet
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        BN_dim,
        hidden_dim,
        layer=8,
        stack=3,
        kernel=3,
        skip=True,
        causal=True,
        dilated=True,
    ):
        super().__init__()

        # input is a sequence of features of shape (B, N, L)
        # here, N/2 refers to the number of frequencies, stacked real and imaginary components

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=EPSILON)
        else:
            self.LN = cLN(input_dim, eps=EPSILON)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=2**i,
                            padding=2**i,
                            skip=skip,
                            causal=causal,
                        )
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=1,
                            padding=1,
                            skip=skip,
                            causal=causal,
                        )
                    )
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += kernel - 1

        # print("receptive field: {:d} time steps".format(self.receptive_field))

        self.output = nn.Conv1d(BN_dim, output_dim, 1)

        self.skip = skip

    def forward(self, input):
        # input shape: (B, N, L)
        # normalization
        output = self.BN(self.LN(input))

        # pass to TCN
        if self.skip:
            skip_connection = 0.0
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output
