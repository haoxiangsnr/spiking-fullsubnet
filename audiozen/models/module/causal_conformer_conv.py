import torch.nn as nn
import torch.nn.functional as F

from audiozen.constant import EPSILON


class CausalConformerConvBlock(nn.Module):
    def __init__(
        self,
        feat_dim,
        kernel_size=32,
        dropout=0.1,
    ):
        """Conformer Convolution Block.

        Args:
            feat_dim: the number of input features.
            kernel_size: the size of the convolving kernel. Defaults to 32 in the paper.
            dropout: the probability of an element to be zeroed. Defaults to 0.1 in the paper.

        Note:
            The input tensor should be in the shape of [B, F, T].
            We will change the layer norm to causal layer norm in the future.
        """
        super().__init__()
        assert kernel_size >= 3, "kernel_size must be larger than 3."

        self.kernel_size = kernel_size
        self.padding_size = kernel_size - 1

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=feat_dim,
            out_channels=feat_dim * 2,  # for GLU
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.pointwise_conv2 = nn.Conv1d(
            in_channels=feat_dim,
            out_channels=feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.depthwise_conv = nn.Conv1d(
            in_channels=feat_dim,
            out_channels=feat_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding_size,
            groups=feat_dim,
            bias=True,
        )

        self.norm = nn.LayerNorm(feat_dim, eps=EPSILON)

        self.activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, xs):
        """Forward.

        Args:
            x: [B, F, T]

        Returns:
            [B, F, T]
        """
        # Layernorm (sentence_length should be in the second dimension)
        input = xs.transpose(1, 2)  # [B, T, F]
        input = self.norm(input)  # [B, T, F]
        input = input.transpose(1, 2)  # [B, F, T]

        # Pointwise Conv
        input = self.pointwise_conv1(input)  # [B, 2 * F, T]

        # GLU
        input = F.glu(input, dim=1)  # [B, F, T]

        # 1D Depthwise Conv
        input = self.depthwise_conv(input)  # [B, F, T]
        input = input[:, :, : -self.padding_size]

        # Layernorm
        input = input.transpose(1, 2)  # [B, T, F]
        input = self.norm(input)  # [B, T, F]

        # Swish Activation
        input = self.activation(input)  # [B, T, F]

        # Pointwise Conv
        input = input.transpose(1, 2)  # [B, F, T]
        input = self.pointwise_conv2(input)  # [B, F, T]

        # Dropout
        input = self.dropout(input)  # [B, F, T]

        return input + xs


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    x = torch.rand(2, 257, 100)
    block = CausalConformerConvBlock(feat_dim=257, kernel_size=3)
    # about 0.28 M and 42 M MACs
    summary(block, input_size=(2, 257, 100), device="cpu")
    y = block(x)
    print(y.shape)
