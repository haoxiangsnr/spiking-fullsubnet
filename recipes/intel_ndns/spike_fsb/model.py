import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_spiking_neuron import LSTMState, efficient_spiking_neuron
from einops import rearrange
from torch import nn
from torch.nn import functional

from audiozen.constant import EPSILON


def deepfiltering(complex_spec, coefs, frame_size: int):
    """Deep filtering implementation using `torch.einsum`. Requires unfolded spectrogram.

    Args:
        spec (complex Tensor): Spectrogram of shape [B, C, F, T, 2]
        coefs (complex Tensor): Coefficients of shape [B, C * frame_size, F, T, 2]

    Returns:
        spec (complex Tensor): Spectrogram of shape [B, C, F, T]
    """
    need_unfold = frame_size > 1

    if need_unfold:
        complex_spec = F.pad(complex_spec, (frame_size - 1, 0))
        complex_spec = complex_spec.unfold(3, frame_size, 1)  # [B, C, F, T, df]
    else:
        complex_spec = complex_spec.unsqueeze(-1)  # [B, C, F, T, 1]

    complex_coefs = torch.complex(coefs[..., 0], coefs[..., 1])  # [B, C, F, T]
    complex_coefs = rearrange(
        complex_coefs, "b (c df) f t -> b c df f t", df=frame_size
    )

    # df
    out = torch.einsum("...ftn,...nft->...ft", complex_spec, complex_coefs)

    return out


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
                LSTMState(
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


class SubBandSequenceWrapper(SequenceModel):
    def __init__(self, df_order, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_order = df_order

    def forward(self, subband_input):
        """Forward pass.

        Args:
            subband_input: the input of shape [B, N, C, F_subband, T]

        Returns:
            output: the output of shape [B, df_order, N * F_subband_out, T, 2]
        """

        (
            batch_size,
            num_subband_units,
            num_channels,
            num_subband_freqs,
            num_frames,
        ) = subband_input.shape
        assert num_channels == 1

        output = rearrange(subband_input, "b n c fs t -> (b n) (c fs) t")
        output, all_layer_outputs = super().forward(output)
        output = rearrange(
            output,
            "(b n) (c fc df) t -> b df (n fc) t c",
            b=batch_size,
            c=num_channels * 2,
            df=self.df_order,
        )

        # e.g., [B, 3, 20, T, 2]

        return output, all_layer_outputs

        # for deep filter
        # [B, df_order, F, T, C]

        # output = subband_input.reshape(
        #     batch_size * num_subband_units, num_subband_freqs, num_frames
        # )
        # output = super().forward(output)

        # # [B, N, C, 2, center, T]
        # output = output.reshape(batch_size, num_subband_units, 2, -1, num_frames)

        # # [B, 2, N, center, T]
        # output = output.permute(0, 2, 1, 3, 4).contiguous()

        # # [B, C, N * F_subband_out, T]
        # output = output.reshape(batch_size, 2, -1, num_frames)

        # return output


class SubbandModel(BaseModel):
    def __init__(
        self,
        freq_cutoffs,
        sb_num_center_freqs,
        sb_num_neighbor_freqs,
        fb_num_center_freqs,
        fb_num_neighbor_freqs,
        sb_df_orders,
        sequence_model,
        hidden_size,
        activate_function=False,
        norm_type="offline_laplace_norm",
        shared_weights=False,
        bn=False,
    ):
        super().__init__()

        sb_models = []
        for (
            sb_num_center_freq,
            sb_num_neighbor_freq,
            fb_num_center_freq,
            fb_num_neighbor_freq,
            df_order,
        ) in zip(
            sb_num_center_freqs,
            sb_num_neighbor_freqs,
            fb_num_center_freqs,
            fb_num_neighbor_freqs,
            sb_df_orders,
        ):
            sb_models.append(
                SubBandSequenceWrapper(
                    df_order=df_order,
                    input_size=(sb_num_center_freq + sb_num_neighbor_freq * 2)
                    + (fb_num_center_freq + fb_num_neighbor_freq * 2),
                    output_size=sb_num_center_freq * 2 * df_order,
                    hidden_size=hidden_size,
                    num_layers=2,
                    sequence_model=sequence_model,
                    bidirectional=False,
                    output_activate_function=activate_function,
                    shared_weights=shared_weights,
                    bn=bn,
                )
            )

        self.sb_models = nn.ModuleList(sb_models)
        self.freq_cutoffs = freq_cutoffs
        self.sb_num_center_freqs = sb_num_center_freqs
        self.sb_num_neighbor_freqs = sb_num_neighbor_freqs
        self.fb_num_center_freqs = fb_num_center_freqs
        self.fb_num_neighbor_freqs = fb_num_neighbor_freqs

        self.norm = self.norm_wrapper(norm_type)

    def _freq_unfold(
        self,
        input: torch.Tensor,
        lower_cutoff_freq=0,
        upper_cutoff_freq=20,
        num_center_freqs=1,
        num_neighbor_freqs=15,
    ):
        """Unfold frequency axis.

        Args:
            input: magnitude spectrogram of shape (batch_size, 1, num_freqs, num_frames).
            cutoff_freq_lower: lower cutoff frequency.
            cutoff_freq_higher: higher cutoff frequency.
            num_center_freqs: number of center frequencies.
            num_neighbor_freqs: number of neighbor frequencies.

        Returns:
            [batch_size, num_subband_units, num_channels, num_subband_freqs, num_frames]

        Note:
            We assume that the num_neighbor_freqs should less than the minimum subband intervel.
        """
        batch_size, num_channels, num_freqs, num_frames = input.shape
        assert num_channels == 1, "Only mono audio is supported."

        if (upper_cutoff_freq - lower_cutoff_freq) % num_center_freqs != 0:
            raise ValueError(
                f"The number of center frequencies should be divisible by the subband freqency interval. "
                f"Got {num_center_freqs=}, {upper_cutoff_freq=}, and {lower_cutoff_freq=}. "
                f"The subband freqency interval is {upper_cutoff_freq - lower_cutoff_freq}."
            )

        # extract valid input with the shape of [batch_size, 1, num_freqs, num_frames]
        if lower_cutoff_freq == 0:
            # lower = 0, upper = upper_cutoff_freq + num_neighbor_freqs
            valid_input = input[..., 0 : (upper_cutoff_freq + num_neighbor_freqs), :]
            valid_input = functional.pad(
                input=valid_input,
                pad=(0, 0, num_neighbor_freqs, 0),
                mode="reflect",
            )

        elif upper_cutoff_freq == num_freqs:
            # lower = lower_cutoff_freq - num_neighbor_freqs, upper = num_freqs
            valid_input = input[
                ..., lower_cutoff_freq - num_neighbor_freqs : num_freqs, :
            ]

            valid_input = functional.pad(
                input=valid_input,
                pad=(0, 0, 0, num_neighbor_freqs),
                mode="reflect",
            )
        else:
            # lower = lower_cutoff_freq - num_neighbor_freqs, upper = upper_cutoff_freq + num_neighbor_freqs
            valid_input = input[
                ...,
                lower_cutoff_freq
                - num_neighbor_freqs : upper_cutoff_freq
                + num_neighbor_freqs,
                :,
            ]

        # unfold
        # [B, C * kernel_size, N]
        subband_unit_width = num_center_freqs + num_neighbor_freqs * 2
        output = functional.unfold(
            input=valid_input,
            kernel_size=(subband_unit_width, num_frames),
            stride=(num_center_freqs, num_frames),
        )
        num_subband_units = output.shape[-1]

        output = output.reshape(
            batch_size,
            num_channels,
            subband_unit_width,
            num_frames,
            num_subband_units,
        )

        # [B, N, C, F_subband, T]
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output

    def forward(self, noisy_input, fb_output):
        """Forward pass.

        Args:
            input: magnitude spectrogram of shape (batch_size, 1, num_freqs, num_frames).
        """
        batch_size, num_channels, num_freqs, num_frames = noisy_input.size()
        assert num_channels == 1, "Only mono audio is supported."

        subband_output = []
        subband_all_layer_outputs = []
        for sb_idx, sb_model in enumerate(self.sb_models):
            if sb_idx == 0:
                lower_cutoff_freq = 0
                upper_cutoff_freq = self.freq_cutoffs[0]
            elif sb_idx == len(self.sb_models) - 1:
                lower_cutoff_freq = self.freq_cutoffs[-1]
                upper_cutoff_freq = num_freqs
            else:
                lower_cutoff_freq = self.freq_cutoffs[sb_idx - 1]
                upper_cutoff_freq = self.freq_cutoffs[sb_idx]

            # unfold frequency axis
            # [B, N, C, F_subband, T]
            noisy_subband = self._freq_unfold(
                noisy_input,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.sb_num_center_freqs[sb_idx],
                self.sb_num_neighbor_freqs[sb_idx],
            )

            # [B, N, C, F_subband, T]
            fb_subband = self._freq_unfold(
                fb_output,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.fb_num_center_freqs[sb_idx],
                self.fb_num_neighbor_freqs[sb_idx],
            )

            sb_model_input = torch.cat([noisy_subband, fb_subband], dim=-2)
            sb_model_input = self.norm(sb_model_input)
            sb_model_output, sb_all_layer_outputs = sb_model(sb_model_input)
            subband_output.append(sb_model_output)
            subband_all_layer_outputs.append(sb_all_layer_outputs)

        # [B, C, F, T]
        # output = torch.cat(subband_output, dim=-2)
        return subband_output, subband_all_layer_outputs


class Separator(BaseModel):
    """_summary_

    Requires:
        einops: install with `pip install einops`

    Args:
        nn: _description_
    """

    def __init__(
        self,
        sr,
        n_fft,
        hop_length,
        win_length,
        fdrc,
        num_freqs,
        freq_cutoffs,
        sb_num_center_freqs,
        sb_num_neighbor_freqs,
        fb_num_center_freqs,
        fb_num_neighbor_freqs,
        fb_hidden_size,
        sb_hidden_size,
        sb_df_orders,
        sequence_model,
        fb_output_activate_function,
        sb_output_activate_function,
        norm_type,
        shared_weights=False,
        bn=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fdrc = fdrc
        self.freq_cutoffs = freq_cutoffs
        self.sb_df_orders = sb_df_orders

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
            shared_weights=shared_weights,
            bn=bn,
        )

        self.sb_model = SubbandModel(
            freq_cutoffs=freq_cutoffs,
            sb_num_center_freqs=sb_num_center_freqs,
            sb_num_neighbor_freqs=sb_num_neighbor_freqs,
            fb_num_center_freqs=fb_num_center_freqs,
            fb_num_neighbor_freqs=fb_num_neighbor_freqs,
            sb_df_orders=sb_df_orders,
            hidden_size=sb_hidden_size,
            sequence_model=sequence_model,
            activate_function=sb_output_activate_function,
            shared_weights=shared_weights,
            bn=bn,
        )

        self.norm = self.norm_wrapper(norm_type)

    def forward(self, y):
        ndim = y.dim()
        assert ndim in (2, 3), "Input must be 2D (B, T) or 3D tensor (B, 1, T)"

        if ndim == 3:
            assert y.size(1) == 1, "Input must be 2D (B, T) or 3D tensor (B, 1, T)"
            y = y.squeeze(1)

        complex_stft = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            return_complex=True,
        )  # [B, F, T]
        complex_stft = complex_stft.unsqueeze(1)

        noisy_mag = torch.abs(complex_stft)  # [B, 1, F, T]

        # ================== Fullband ==================
        noisy_mag = noisy_mag**self.fdrc  # fdrc
        noisy_mag = noisy_mag[..., :-1, :]  # [B, 1, F, T]
        fb_input = rearrange(self.norm(noisy_mag), "b c f t -> b (c f) t")
        fb_output, fb_all_layer_outputs = self.fb_model(fb_input)  # [B, F, T]
        fb_output = rearrange(fb_output, "b f t -> b 1 f t")

        # ================== Subband ==================
        # list [[B, df, F_1, T, 2], [B, df, F_2, T, 2], ...]
        df_coefs_list, sb_all_layer_outputs = self.sb_model(noisy_mag, fb_output)

        # ================== Iterative Masking ==================
        num_filtered = 0
        enhanced_spec_list = []
        for df_coefs, df_order in zip(df_coefs_list, self.sb_df_orders):
            # [B, C, F , T] = [B, C, F, ]
            num_sub_freqs = df_coefs.shape[2]
            complex_stft_in = complex_stft[
                ..., num_filtered : num_filtered + num_sub_freqs, :
            ]
            enhanced_subband = deepfiltering(
                complex_stft_in, df_coefs, frame_size=df_order
            )  # [B, 1, F, T] of complex
            enhanced_spec_list.append(enhanced_subband)
            num_filtered += num_sub_freqs

        # [B, C, F, T]
        enhanced_spec = torch.cat(enhanced_spec_list, dim=-2)

        enhanced_stft = complex_stft.clone()  # [B, C, F， T]

        enhanced_stft[..., :-1, :] = enhanced_spec
        enhanced_stft = enhanced_stft.squeeze(1)  # [B, F, T]

        # Magnitude
        enhanced_mag = torch.abs(enhanced_stft)  # [B, F, T]

        enhanced_y = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            length=y.size(-1),
        )
        return enhanced_y, enhanced_mag, fb_all_layer_outputs, sb_all_layer_outputs


if __name__ == "__main__":
    import toml
    from torchinfo import summary

    model = Separator(
        sr=16000,
        fdrc=0.5,
        n_fft=512,
        sb_df_orders=[5, 3, 1, 1],
        hop_length=256,
        win_length=512,
        num_freqs=256,
        sequence_model="GSU",
        fb_hidden_size=320,
        fb_output_activate_function=False,
        freq_cutoffs=[32, 128, 192],
        sb_num_center_freqs=[4, 32, 64, 64],
        sb_num_neighbor_freqs=[15, 15, 15, 15],
        fb_num_center_freqs=[4, 32, 64, 64],
        fb_num_neighbor_freqs=[0, 0, 0, 0],
        sb_hidden_size=160,
        sb_output_activate_function=False,
        norm_type="offline_laplace_norm",
        shared_weights=True,
        bn=True,
    )
    noisy_y = torch.rand(5, 16400)
    noisy_y = noisy_y.to("cuda:3")
    model = model.to("cuda:3")
    enhanced, enhanced_mag, fb_all_layer_outputs, sb_all_layer_outputs = model(noisy_y)
    print(enhanced.shape, enhanced_mag.shape)
    # for i in range(len(fb_all_layer_outputs)):
    #     print(fb_all_layer_outputs[i].size())
    # for i in range(len(sb_all_layer_outputs)):
    #     print(i)
    #     for j in range(len(sb_all_layer_outputs[i])):
    #         print(sb_all_layer_outputs[i][j].size())
    # print(fb_all_layer_outputs)
    # print(model(noisy_y).shape)
    # summary(model, input_data=(noisy_y,), device="cpu")
