from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor, jit


class MogrifierLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, mogrify_steps):
        super().__init__()
        self.mogrify_steps = mogrify_steps

        self.lstm = nn.LSTMCell(input_size, hidden_size)

        self.mogrifier_list = nn.ModuleList(
            [
                nn.Linear(hidden_size, input_size),  # q
                nn.Linear(input_size, hidden_size),  # r
                nn.Linear(hidden_size, input_size),  # q
                nn.Linear(input_size, hidden_size),  # r
                nn.Linear(hidden_size, input_size),  # q
            ]
        )

    @jit.script_method
    def mogrify(self, x, h):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        x = (2 * torch.sigmoid(self.mogrifier_list[0](h))) * x
        h = (2 * torch.sigmoid(self.mogrifier_list[1](x))) * h
        x = (2 * torch.sigmoid(self.mogrifier_list[2](h))) * x
        h = (2 * torch.sigmoid(self.mogrifier_list[3](x))) * h
        x = (2 * torch.sigmoid(self.mogrifier_list[4](h))) * x

        return x, h

    @jit.script_method
    def forward(self, x, states):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, (ht, ct)


class MogrifierLSTMLayer(jit.ScriptModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.cell = MogrifierLSTMCell(*args, **kwargs)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        input = input.transpose(0, 1)  # [T, B, F]

        outputs = []
        for i in range(len(input)):
            out, state = self.cell(input[i], state)
            outputs += [out]

        outputs = torch.stack(outputs, dim=0)  # [T, B, F]
        outputs = outputs.transpose(0, 1).contiguous()  # [B, T, F]
        return outputs, state


class MogrifierLSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, num_layers, mogrify_steps=4) -> None:
        super().__init__()

        layers = [MogrifierLSTMLayer(input_size, hidden_size, mogrify_steps)]
        for _ in range(num_layers - 1):
            layers.append(MogrifierLSTMLayer(hidden_size, hidden_size, mogrify_steps))

        self.layers = nn.ModuleList(layers)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])

        output = input
        h_0, c_0 = states

        for i, layer in enumerate(self.layers):
            in_state = (h_0[i], c_0[i])  # [B, H]
            output, out_state = layer(output, in_state)  # [B, T, H]
            output_states += [out_state]

        return output, output_states


class MogrifierLSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, mogrify_steps=4):
        super().__init__()

        self.mogrifier_lstm = MogrifierLSTM(
            input_size, hidden_size, num_layers, mogrify_steps=mogrify_steps
        )
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input, states=None):
        batch_size = input.shape[0]

        if states is None:
            h_0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
            c_0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
            states = (h_0, c_0)

        output, output_states = self.mogrifier_lstm(input, states)

        return output, output_states


if __name__ == "__main__":
    model = MogrifierLSTMWrapper(
        input_size=257,
        hidden_size=512,
        num_layers=2,
        mogrify_steps=4,
    )

    input = torch.rand(1, 100, 257)
    output, states = model(input)
    print(output.shape)
