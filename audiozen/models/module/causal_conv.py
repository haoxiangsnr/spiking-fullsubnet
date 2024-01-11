import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from audiozen.constant import EPSILON


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """

        Args:
            num_inputs:
            num_channels: The list of all channels?
            kernel_size:
            dropout:

        Inputs: x
            - x: x has a dimension of [B, C, T]
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_activate_function, **kwargs):
        """

        Args:
            in_channels:
            out_channels:
            encoder_activate_function:
            **kwargs:
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1),
            **kwargs,  # 这里不是左右 pad，而是上下 pad 为 0，左右分别 pad 1...
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = getattr(nn, encoder_activate_function)()

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalConv2DBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm([1, 481, 100]),
            nn.Conv2d(in_channels=1, out_channels=481, kernel_size=1),
        )

    def forward(self, x):
        pass


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels=257,
        hidden_channel=512,
        out_channels=257,
        kernel_size=3,
        dilation=1,
        use_skip_connection=True,
        causal=False,
    ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channel, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=EPSILON)
        padding = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        self.depthwise_conv = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel_size=kernel_size,
            stride=1,
            groups=hidden_channel,
            padding=padding,
            dilation=dilation,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=EPSILON)
        self.sconv = nn.Conv1d(hidden_channel, out_channels, 1)
        # self.tcn_block = nn.Sequential(
        #     nn.Conv1d(in_channels, hidden_channel, 1),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=EPSILON),
        #     nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1,
        #               groups=hidden_channel, padding=padding, dilation=dilation, bias=True),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=EPSILON),
        #     nn.Conv1d(hidden_channel, out_channels, 1)
        # )

        self.causal = causal
        self.padding = padding
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        """
        x: [channels, T]
        """
        if self.use_skip_connection:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, : -self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return x + output
        else:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, : -self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return output


class STCNBlock(nn.Module):
    def __init__(
        self,
        in_channels=257,
        hidden_channel=512,
        out_channels=257,
        kernel_size=3,
        dilation=1,
        use_skip_connection=True,
        causal=False,
    ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channel, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=EPSILON)
        padding = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        self.depthwise_conv = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel_size=kernel_size,
            stride=1,
            groups=hidden_channel,
            padding=padding,
            dilation=dilation,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=EPSILON)
        self.sconv = nn.Conv1d(hidden_channel, out_channels, 1)
        # self.tcn_block = nn.Sequential(
        #     nn.Conv1d(in_channels, hidden_channel, 1),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=EPSILON),
        #     nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1,
        #               groups=hidden_channel, padding=padding, dilation=dilation, bias=True),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=EPSILON),
        #     nn.Conv1d(hidden_channel, out_channels, 1)
        # )

        self.causal = causal
        self.padding = padding
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        """
        x: [channels, T]
        """
        if self.use_skip_connection:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, : -self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return x + output
        else:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, : -self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return output


if __name__ == "__main__":
    a = torch.rand(2, 1, 19, 200)
    l1 = CausalConvBlock(
        1,
        20,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(0, 1),
    )
    l2 = CausalConvBlock(
        20,
        40,
        kernel_size=(3, 2),
        stride=(1, 1),
        padding=1,
    )
    l3 = CausalConvBlock(
        40,
        40,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(0, 1),
    )
    l4 = CausalConvBlock(
        40,
        40,
        kernel_size=(3, 2),
        stride=(1, 1),
        padding=1,
    )
    print(l1(a).shape)
    print(l4(l3(l2(l1(a)))).shape)
