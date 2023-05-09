import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from audiozen.models.base_model import BaseModel


class ResidualLSTM(BaseModel):
    def __init__(self, input_size, output_size, hidden_size, use_activation=True):
        super().__init__()
        self.in_conv = nn.Conv1d(input_size, hidden_size, 1, bias=False)
        self.lstm_1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        self.use_activation = use_activation
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(output_size)

    def forward(self, input):
        output = self.in_conv(input)

        output = rearrange(output, "b f t -> b t f")
        output = output + self.norm_1(self.lstm_1(output)[0])
        output = output + self.norm_1(self.lstm_2(output)[0])
        output = self.norm_2(self.linear(output))

        if self.use_activation:
            output = F.relu(output)

        output = rearrange(output, "b t f -> b f t")

        return output


if __name__ == "__main__":
    model = ResidualLSTM(257, 257, 512)
    input = torch.rand(1, 257, 100)
    output = model(input)
    print(output.shape)
