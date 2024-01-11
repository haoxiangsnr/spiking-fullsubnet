from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from audiozen.acoustics.audio_feature import istft, stft
from recipes.intel_ndns.spiking_fullsubnet.efficient_spiking_neuron import MemoryState, efficient_spiking_neuron


class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        sequence_model="GSN",
        proj_size=0,
        shared_weights=False,
        output_activate_function=None,
        bn=False,
        use_pre_layer_norm=True,
    ):
        super().__init__()

        if use_pre_layer_norm:
            self.pre_layer_norm = nn.LayerNorm(input_size)

        if sequence_model == "GSN":
            self.sequence_model = efficient_spiking_neuron(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                shared_weights=shared_weights,
                bn=bn,
            )
        elif sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
            )
        else:
            raise NotImplementedError(f"Sequence model {sequence_model} not implemented.")

        if proj_size > 0:
            self.proj = nn.Linear(hidden_size, proj_size)
        else:
            self.proj = nn.Identity()

        if output_activate_function == "tanh":
            self.output_activate_function = nn.Tanh()
        elif output_activate_function == "sigmoid":
            self.output_activate_function = nn.Sigmoid()
        elif output_activate_function == "relu":
            self.output_activate_function = nn.ReLU()
        else:
            self.output_activate_function = nn.Identity()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_pre_layer_norm = use_pre_layer_norm
        self.sequence_model_name = sequence_model

    def forward_lstm(self, input):
        assert input.ndim == 3, f"Input tensor must be 3D, but got {input.ndim}D."
        input = rearrange(input, "b f t -> b t f")

        if self.use_pre_layer_norm:
            input = self.pre_layer_norm(input)

        output, _ = self.sequence_model(input)
        output = self.proj(output)
        output = rearrange(output, "b t f -> b f t")
        output = self.output_activate_function(output)
        return output, []

    def forward(self, input):
        """Forward function.

        Args:
            input (`torch.Tensor` of shape `(batch_size, num_freq, sequence_length)` or `(batch_size, num_channels, num_freq, sequence_length)`):
                Input 3D or 4D tensor.

        Returns:
            output (`torch.Tensor` of shape `(batch_size, num_freq, sequence_length)`): Output tensor.
                Output tensor.
        """
        if self.sequence_model_name == "LSTM":
            return self.forward_lstm(input)

        assert input.ndim == 3, f"Input tensor must be 3D, but got {input.ndim}D."

        batch_size, num_freqs, sequence_length = input.shape

        # Initialize memory states.
        states = [
            MemoryState(
                torch.zeros(batch_size, self.hidden_size, device=input.device),
                torch.zeros(batch_size, self.hidden_size, device=input.device),
            )
            for _ in range(self.num_layers)
        ]

        input = rearrange(input, "b f t -> t b f")

        # Apply layer normalization.
        if self.use_pre_layer_norm:
            input = self.pre_layer_norm(input)

        # Pass through the sequence model.
        output, _, all_layer_outputs = self.sequence_model(input, states)

        # Project the output if necessary.
        output = self.proj(output)
        all_layer_outputs += [output]

        # Apply the output activation function.
        output = self.output_activate_function(output)

        output = rearrange(output, "t b f -> b f t")
        return output, all_layer_outputs


def deepfiltering(complex_spec, coef, order: int, num_spks: int):
    """Deep filtering implementation using `torch.einsum`. Requires unfolded spectrogram.

    Args:
        complex_spec (`torch.ComplexTensor` of shape `[B, C, F, T]`):
            Complex spectrogram.
        coef (`torch.Tensor` of shape `[B, C * order, F, T, 2]`):
            Coefficients of the deep filter.
        order (`int`): Order of the deep filter.
        num_spks (`int`): Number of speakers.

    Returns:
        spec (complex Tensor): Spectrogram of shape `[B, C, S, F, T]`.
    """
    need_unfold = order > 1

    if need_unfold:
        complex_spec = F.pad(complex_spec, (order - 1, 0))
        complex_spec = complex_spec.unfold(3, order, 1)  # [B, C, F, T, df]
    else:
        complex_spec = complex_spec.unsqueeze(-1)  # [B, C, F, T, 1]

    complex_spec = complex_spec.unsqueeze(-1)  # [B, C, F, T, df, 1]
    complex_spec = complex_spec.repeat(1, 1, 1, 1, 1, num_spks)  # [B, C, F, T, df, s]

    complex_coef = torch.complex(coef[..., 0], coef[..., 1])  # [B, C * df, s, F, T]
    complex_coef = rearrange(complex_coef, "b (c df) s f t -> b c df s f t", df=order)

    # [B, C, S, F, T]
    out = torch.einsum("...ftds,...dsft->...sft", complex_spec, complex_coef)

    return out


class Model(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        fdrc,
        input_size,
        hidden_size,
        num_layers,
        proj_size,
        output_activate_function,
        df_order,
        use_pre_layer_norm_fb=True,
        bn=False,
        shared_weights=False,
        sequence_model="LSTM",
        num_spks=2,
    ):
        super().__init__()

        self.fb_model = SequenceModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            shared_weights=shared_weights,
            sequence_model=sequence_model,
            proj_size=proj_size * num_spks * df_order * 2,
            output_activate_function=output_activate_function,
            bn=bn,
            use_pre_layer_norm=use_pre_layer_norm_fb,
        )

        self.stft = partial(stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.istft = partial(istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        self.fb_input_size = input_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fdrc = fdrc
        self.df_order = df_order
        self.num_spks = num_spks

    def forward(self, input):
        """Forward function.

        Args:
            input (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Waveform tensor.

        Returns:
            output (`torch.Tensor` of shape `(batch_size, num_freq, sequence_length)`):
                Output tensor.
        """
        assert input.ndim == 2, f"Input tensor must be 2D, but got {input.ndim}D."
        batch_size, sequence_length = input.shape

        noisy_mag, _, noisy_real, noisy_imag = self.stft(input)
        noisy_cmp = torch.complex(real=noisy_real, imag=noisy_imag)
        noisy_cmp = rearrange(noisy_cmp, "b f t -> b 1 f t")

        # ================== Fullband model ==================
        noisy_mag = rearrange(noisy_mag, "b f t -> b 1 f t")
        noisy_mag = noisy_mag**self.fdrc
        fb_input = rearrange(noisy_mag, "b c f t -> b (c f) t")
        fb_output, *_ = self.fb_model(fb_input)
        # `c` is corresponding to complex number
        df_coef = rearrange(fb_output, "b (c d s f) t -> b d s f t c", c=2, d=self.df_order, s=self.num_spks)

        # ================== Reconstruct the output ==================
        enh_stft = deepfiltering(noisy_cmp, df_coef, self.df_order, self.num_spks)  # [B, c, s, f, t]

        if self.num_spks > 1:
            enh_stft = rearrange(enh_stft, "b 1 s f t -> (b s) f t")
            enh_y = self.istft(enh_stft, length=sequence_length)
            enh_y = rearrange(enh_y, "(b s) t -> b s t", s=self.num_spks)
            return enh_y, _
        else:
            enh_stft = rearrange(enh_stft, "b 1 1 f t -> b f t")
            enh_mag = torch.abs(enh_stft)  # For computing DNSMOS loss
            enh_y = self.istft(enh_stft, length=sequence_length)
            return enh_y, enh_mag


if __name__ == "__main__":
    model = Model(
        n_fft=512,
        hop_length=128,
        win_length=512,
        fdrc=0.5,
        input_size=257,
        hidden_size=256,
        num_layers=2,
        proj_size=257,
        output_activate_function=None,
        df_order=3,
        use_pre_layer_norm_fb=True,
        bn=False,
        shared_weights=False,
        sequence_model="LSTM",
        num_spks=2,
    )

    input = torch.rand(2, 16000)
    output = model(input)
    print(output[0].shape)
