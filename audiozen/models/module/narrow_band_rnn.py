from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class SubBandLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, sub_band_size, alpha):
        super(SubBandLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sub_band_size = sub_band_size

        self.weight_x = Parameter(torch.randn(4 * hidden_size, sub_band_size))
        self.weight_h = Parameter(torch.randn(4 * hidden_size, alpha * sub_band_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            input: [B, N, F]
            state: ([B, aN], [B, H])

        Notes:
            input: [B, N, F]
            hidden_state: (h_x: [B, aN], c_x: [B, aN]), 每个样本都会有一个自己的 hidden state
            x_t: [B, F, N]
            W_x: [H, N]
            W_h: [H, H]


        Returns:
            [B, H], ([B, H], [B, H])
        """
        hx, cx = state
        batch_size, num_freqs, sub_band_size = input.shape

        # W_x @ x_t
        # [B, F, N] @ [N, H] => [B, F, H]
        # [N, H] 对于所有频带均共享
        # [B, F, N] => [BF, N]
        input = input.reshape(batch_size * num_freqs, sub_band_size)
        # [BF, N] @ [N, H] = [BF, H], 这里的 H 包含 4 个部分，但是每个部分是对于全频带共享权重的，维度为 [4H, N]
        torch.mm(input, self.weight_x.t())

        # W_h @ h_{t-1}
        # [B, aN] => [B, F, aN]
        # [B, F, aN] @ [aN, H] = [B, H]
        hx = hx[:, None, :].expand(-1, num_freqs, -1).reshape(batch_size * num_freqs, self.alpha * self.sub_band_size)

        gates = torch.mm(input, self.weight_x.t()) + self.bias_ih + torch.mm(hx, self.weight_h.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cx = cx[:, None, :].expand(-1, num_freqs, -1).reshape(batch_size * num_freqs, -1)

        # 这里的 cx 限制了，难道 cx 单独修改一下？
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        hy = hy.reshape(batch_size, num_freqs, self.alpha * self.sub_band_size)
        cy = cy.reshape(batch_size, num_freqs, -1)

        return hy, (hy, cy)


class SubBandGRUCell(nn.Module):
    def __init__(self, input_size: int, sub_band_size: int, hidden_size: int, alpha: int):
        """

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            sub_band_size: The number of features in each frequency
            alpha: The value of

        Notations:
            - H: hidden_state
            - ...

        Inputs: input, hidden_state
            - input of shape [B, F, N]: tensor containing input features
            - hidden_state of shape [B, F, aN]: ...

        Output: new_hidden_state
            - new_hidden_state of shape [B, F, aN]

        Notes:
            - [H, aN] 对于每个样本，每个频带都是共享的，但是每个频带都有它自己的输出（隐状态），这个可不是共享的
        """
        super().__init__()
        # for reset_gate and update gate
        self.weight_x = Parameter(torch.randn(2 * hidden_size, input_size))  # first term: W_x @ x_t
        self.weight_h = Parameter(torch.randn(2 * hidden_size, alpha * sub_band_size))  # second term: W_h @ h_{t-1}
        self.bias_x = Parameter(torch.randn(2 * hidden_size))
        self.bias_h = Parameter(torch.randn(2 * hidden_size))

        # for new gate
        self.weight_x_new_gate = Parameter(torch.randn(hidden_size, input_size))
        self.weight_h_new_gate = Parameter(torch.randn(hidden_size, alpha * sub_band_size))
        self.bias_x_new_gate = Parameter(torch.randn(hidden_size))
        self.bias_h_new_gate = Parameter(torch.randn(hidden_size))

        # from [B, H] to [B, aN]
        # e.g. [B, 256] to [B, 3 * 30]
        self.linear_update_gate = nn.Linear(hidden_size, alpha * sub_band_size, bias=False)
        self.linear_new_gate = nn.Linear(hidden_size, alpha * sub_band_size, bias=False)

        # export useful vars
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sub_band_size = sub_band_size

    def forward(self, input: Tensor, hidden_state: Tensor) -> Tensor:
        batch_size, num_freqs, sub_band_size = input.shape

        input = input.reshape(batch_size * num_freqs, sub_band_size)
        hidden_state = hidden_state.reshape(batch_size * num_freqs, self.alpha * self.sub_band_size)
        print(input.shape, hidden_state.shape)

        # reset_gate and update_gate
        # first "mm" term (W_x @ x_t): [BF, H] @ [N, H] => [BF, H]. [N, H] 对于所有频带均共享
        # second "mm" term (W_h @ h_{t-1}): [BF, aN] @ [aN, H] => [BF, H]
        # H 为存储了特征的变换信息，aN 存储了特征的记忆信息。
        # 特征的记忆信息可能并不需要那么多。我们将其从 H 变为 aN，与子带的大小相关，进而降低计算量。
        gates = (
            torch.mm(input, self.weight_x.t()) + self.bias_x + torch.mm(hidden_state, self.weight_h.t()) + self.bias_h
        )
        reset_gate, update_gate = gates.chunk(2, 1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # new gate
        new_gate = torch.tanh(
            torch.mm(input, self.weight_x_new_gate.t())
            + self.bias_x_new_gate
            + reset_gate * torch.mm(hidden_state, self.weight_h_new_gate.t())
            + self.bias_h_new_gate
        )

        # new hidden state
        # linear_layer used to rescale H to aN
        new_hidden_state = (1 - self.linear_update_gate(update_gate)) * self.linear_new_gate(
            new_gate
        ) + self.linear_update_gate(update_gate) * hidden_state

        new_hidden_state = new_hidden_state.reshape(batch_size, num_freqs, self.alpha * self.sub_band_size)

        # [B, aN]
        return new_hidden_state


class SubBandGRULayer(nn.Module):
    def __init__(self, input_size: int, sub_band_size: int, hidden_size: int, alpha: int):
        """

        Args:
            input_size:
            sub_band_size:
            hidden_size:
            alpha:

        Inputs: input, hidden_state
            - input of shape [B, F, N, T]
            - input of shape [B, aN]
        """
        super().__init__()
        # export useful vars
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sub_band_size = sub_band_size

        self.cell = SubBandGRUCell(input_size, sub_band_size, hidden_size, alpha)

    def forward(self, input, hidden_state=None):
        batch_size, feature_size, sub_band_size, seq_len = input.shape
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)

        outputs = []
        for i in range(seq_len):
            hidden_state = self.cell(input[..., i], hidden_state)
            outputs.append(hidden_state)

        outputs = torch.stack(outputs, dim=-1)

        return outputs, hidden_state


class SubBandGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        sub_band_size: int,
        hidden_size: int,
        alpha: int,
        num_layers: int,
    ):
        """

        Args:
            input_size:
            sub_band_size:
            hidden_size:
            alpha:
            num_layers:

        Inputs: input, hidden_state
            - input of shape [B, F, N, T]
            - input of shape [B, aN]

        Outputs:
            hidden_state: [num_layers, B, aN]
        """
        super().__init__()
        # export useful vars
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sub_band_size = sub_band_size
        self.num_layers = num_layers

        self.layers = [
            SubBandGRULayer(input_size, sub_band_size, hidden_size, alpha),
        ]
        self.layers += [
            SubBandGRULayer(sub_band_size * alpha, sub_band_size, sub_band_size * alpha, alpha)
            for _ in range(num_layers - 1)
        ]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input, hidden_state=None):
        batch_size, feature_size, sub_band_size, seq_len = input.shape

        if hidden_state is None:
            hidden_state = torch.zeros(
                self.num_layers,
                batch_size,
                feature_size,
                self.alpha * self.sub_band_size,
                dtype=input.dtype,
                device=input.device,
            )

        i = 0
        output = input
        output_states = []
        for layer in self.layers:
            print("input", i, output.shape, hidden_state[i].shape)
            output, output_state = layer(output, hidden_state[i])
            print("output", i, output.shape, hidden_state[i].shape)
            output_states += [output_state]
            i += 1

        output_states = torch.stack(output_states, dim=0)
        return output, output_states


if __name__ == "__main__":
    batch_size = 2
    num_freqs = 257
    hidden_size = 256
    sub_band_size = 3
    alpha = 10
    num_frames = 10
    num_layers = 3

    ipt = torch.rand(batch_size, num_freqs, sub_band_size, num_frames)
    hx = torch.rand(num_layers, batch_size, num_freqs, sub_band_size * alpha)
    model = SubBandGRU(
        input_size=sub_band_size,
        sub_band_size=sub_band_size,
        hidden_size=hidden_size,
        alpha=alpha,
        num_layers=num_layers,
    )
    print(model(ipt)[0].shape)
