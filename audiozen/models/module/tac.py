import torch
import torch.nn as nn
from einops import pack, rearrange, reduce, repeat
from torch import Tensor

from audiozen.constant import EPSILON


class TransformAverageConcatenate(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()

        self.transform_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
        )
        self.average_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, input_size),
            nn.GELU(),
        )

        self.norm_layer = nn.GroupNorm(1, input_size, eps=EPSILON)

    def forward(self, input: Tensor):
        batch_size, num_groups, num_freqs, num_frames = input.shape

        output = rearrange(input, "B G F T -> (B G) F T")
        output = self.norm_layer(output)
        output = rearrange(output, "(B G) F T -> B G F T", B=batch_size)

        # Transform
        group_input = rearrange(output, "B G F T -> (B G T) F")
        group_output = self.transform_layer(group_input)
        group_output = rearrange(
            group_output, "(B G T) F -> B T G F", B=batch_size, G=num_groups
        )

        # Average
        group_mean = reduce(group_output, "B T G F -> B T F", "mean")
        group_mean = rearrange(group_mean, "B T F -> (B T) F")
        group_mean = self.average_layer(group_mean)
        group_mean = repeat(group_mean, "BT F -> BT G F", G=num_groups)

        # Concatenate
        group_output = rearrange(group_output, "B T G F -> (B T) G F")
        group_output, _ = pack([group_output, group_mean], "BT G *")
        group_output = self.output_layer(group_output)  # [BT, G, F]
        group_output = rearrange(group_output, "(B T) G F -> B G F T", B=batch_size)

        return input + group_output


if __name__ == "__main__":
    input = torch.rand(2, 4, 64, 256)
    model = TransformAverageConcatenate(64, 256)
    output = model(input)
    print(output.shape)
