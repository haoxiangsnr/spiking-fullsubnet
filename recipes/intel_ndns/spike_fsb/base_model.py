import torch
import torch.nn as nn
from efficient_spiking_neuron import MemoryState, efficient_spiking_neuron

from audiozen.constant import EPSILON


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def offline_laplace_norm(input, return_mu=False):
        """Normalize the input with the utterance-level mean.

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]

        Notes:
            As mentioned in the paper, the offline normalization is used.
            Based on a bunch of experiments, the offline normalization have the same performance as the cumulative one and have a faster convergence than the cumulative one.
            Therefore, we use the offline normalization as the default normalization method.
        """
        # utterance-level mu
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = input / (mu + EPSILON)

        if return_mu:
            return normed, mu
        else:
            return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """Normalize the input with the cumulative mean

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device,
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(
            batch_size * num_channels, 1, num_frames
        )

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)
        std = torch.std(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = (input - mu) / (std + EPSILON)
        return normed

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        else:
            raise NotImplementedError(
                "You must set up a type of Norm. "
                "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc."
            )
        return norm


class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        bidirectional,
        sequence_model="GRU",
        output_activate_function="Tanh",
        num_groups=4,
        mogrify_steps=5,
        dropout=0.0,
        shared_weights=False,
        bn=False,
    ):
        super().__init__()
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        elif sequence_model == "GSU":
            # print(f"input_size: {input_size}, hidden_size: {hidden_size}, num_layers: {num_layers}, output_size: {output_size}")
            self.sequence_model = efficient_spiking_neuron(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                shared_weights=shared_weights,
                bn=bn,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if int(output_size):
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )

        self.output_activate_function = output_activate_function
        self.output_size = output_size

        # only for custom_lstm and are needed to clean up
        self.sequence_model_name = sequence_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3, f"Shape is {x.shape}."
        batch_size, _, _ = x.shape

        if self.sequence_model_name == "LayerNormLSTM":
            states = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                )
                for _ in range(self.num_layers)
            ]
        elif self.sequence_model_name == "GSU":
            states = [
                MemoryState(
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                )
                for _ in range(self.num_layers)
            ]
        else:
            states = None

        x = x.permute(2, 0, 1).contiguous()  # [B, F, T] => [T, B, F]
        if (
            self.sequence_model_name == "LSTM"
            or self.sequence_model_name == "GRU"
            or self.sequence_model_name == "GSU"
        ):
            assert self.sequence_model_name == "GSU"
            if self.sequence_model_name == "GSU":
                o, _, all_layer_outputs = self.sequence_model(x, states)
            else:
                o, _ = self.sequence_model(x, states)  # [T, B, F] => [T, B, F]
            if self.output_size:
                o = self.fc_output_layer(o)  # [T, B, F] => [T, B, F]
                all_layer_outputs += [o]
            if self.output_activate_function:
                o = self.activate_function(o)
        elif self.sequence_model_name == "LIF":
            o = self.sequence_model(x)

        return (
            o.permute(1, 2, 0).contiguous(),
            all_layer_outputs,
        )  # [T, B, F] => [B, F, T]
