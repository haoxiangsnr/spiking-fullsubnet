import math
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import Parameter


MemoryState = namedtuple("MemoryState", ["hx", "cx"])


def efficient_spiking_neuron(
    input_size,
    hidden_size,
    num_layers,
    shared_weights=False,
    bn=False,
    batch_first=False,
):
    """
    Instantiate efficient spiking networks where each spiking neuron uses the gating mechanism to control the decay of membrane potential.
    :param input_size:
    :param hidden_size:
    :param num_layers:
    :param shared_weights: whether weights of the gate are shared with the ones of the cell or not.
    :param bn: whether batchnorm is used or not.
    :param batch_first: Not used.
    :return:
    """
    #     # The following are not implemented.
    assert not batch_first
    # assert shared_weights
    # assert bn

    return StackedGSU(
        num_layers,
        GSULayer,
        first_layer_args=[GSUCell, input_size, hidden_size, shared_weights, bn],
        other_layer_args=[GSUCell, hidden_size, hidden_size, shared_weights, bn],
    )


class StackedGSU(nn.Module):
    # __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedGSU, self).__init__()
        self.layers = init_stacked_gsu(num_layers, layer, first_layer_args, other_layer_args)

    def forward(self, input, states):
        output_states = []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        all_layer_output = [input]
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            all_layer_output += [output]
            i += 1
        return output, output_states, all_layer_output


def init_stacked_gsu(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class GSULayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(GSULayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class Triangle(torch.autograd.Function):
    """Spike firing activation function"""

    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class GSUCell(nn.Module):
    def __init__(self, input_size, hidden_size, shared_weights=False, bn=False):
        super(GSUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.shared_weights = shared_weights
        self.use_bn = bn
        if self.shared_weights:
            self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        else:
            self.weight_ih = Parameter(torch.empty(2 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(2 * hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()

        # self.scale_factor = Parameter(torch.ones(hidden_size))
        if self.use_bn:
            self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx, cx = state
        if self.shared_weights:
            weight_ih = self.weight_ih.repeat((2, 1))
            weight_hh = self.weight_hh.repeat((2, 1))
        else:
            weight_ih = self.weight_ih
            weight_hh = self.weight_hh
        gates = (
            torch.mm(input, weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )
        forgetgate, cellgate = gates.chunk(2, 1)
        forgetgate = torch.sigmoid(forgetgate)
        cy = forgetgate * cx + (1 - forgetgate) * cellgate
        if self.use_bn:
            cy = self.batchnorm(cy)
        hy = Triangle.apply(cy)  # replace the Tanh activation function with step function to ensure binary outputs.

        return hy, (hy, cy)


if __name__ == "__main__":
    input_size = 256
    hidden_size = 320
    num_layers = 2
    shared_weights = True
    bn = True
    batch_size = 128
    T = 100
    x = torch.rand((batch_size, input_size, T))  # [B, F, T]
    sequence_model = efficient_spiking_neuron(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        shared_weights=shared_weights,
        bn=bn,
    )

    states = [
        MemoryState(
            torch.zeros(batch_size, hidden_size, device=x.device),
            torch.zeros(batch_size, hidden_size, device=x.device),
        )
        for _ in range(num_layers)
    ]
    x = x.permute(2, 0, 1).contiguous()  # [B, F, T] => [T, B, F]
    o, _ = sequence_model(x, states)  # [T, B, F] => [T, B, F]
