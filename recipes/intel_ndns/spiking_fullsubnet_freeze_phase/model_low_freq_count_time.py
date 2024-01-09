import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_spiking_neuron import MemoryState, efficient_spiking_neuron
from einops import rearrange
from torch import nn
from torch.nn import functional

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.constant import EPSILON
from audiozen.metric import compute_neuronops


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
        sequence_model="GSU",
        output_activate_function="Tanh",
        num_groups=4,
        mogrify_steps=5,
        dropout=0.0,
        shared_weights=False,
        bn=False,
    ):
        super().__init__()
        if sequence_model == "GSU":
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

        if self.sequence_model_name == "GSU":
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
        if self.sequence_model_name == "GSU":
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
        shape = input.shape
        *_, num_freqs, num_frames = input.shape
        input = input.reshape(-1, num_freqs, num_frames)

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
        cumulative_mean = cumulative_mean.reshape(-1, 1, num_frames)

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(*shape[:-2], num_freqs, num_frames)

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
            start_sb = time.time()

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

            after_sb_norm = time.time()

            sb_model_output, sb_all_layer_outputs = sb_model(sb_model_input)
            subband_output.append(sb_model_output)
            subband_all_layer_outputs.append(sb_all_layer_outputs)

            print(
                f"bottleneck_{sb_idx}: {(after_sb_norm - start_sb) / (16000 / 128) * 1000}ms"
            )

        # [B, C, F, T]
        # output = torch.cat(subband_output, dim=-2)
        return subband_output, subband_all_layer_outputs


class Separator(BaseModel):
    def __init__(
        self,
        sr,
        n_fft,
        hop_length,
        win_length,
        fdrc,
        num_freqs,
        fb_freqs,  # number of low frequency bins extracted from fullband model input.
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
        self.num_repeats = num_freqs // fb_freqs
        self.fb_freqs = fb_freqs

        self.fb_model = SequenceModel(
            input_size=fb_freqs,
            output_size=fb_freqs,
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
            norm_type=norm_type,
        )

        self.stft = partial(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        self.istft = partial(
            istft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        self.norm = self.norm_wrapper(norm_type)

    def forward(self, noisy_y):
        ndim = noisy_y.dim()
        assert ndim in (2, 3), "Input must be 2D (B, T) or 3D tensor (B, 1, T)"

        if ndim == 3:
            assert (
                noisy_y.size(1) == 1
            ), "Input must be 2D (B, T) or 3D tensor (B, 1, T)"
            noisy_y = noisy_y.squeeze(1)

        start = time.time()
        noisy_mag, _, noisy_real, noisy_imag = self.stft(noisy_y)
        complex_stft = torch.complex(noisy_real, noisy_imag)  # [B, F, T]
        complex_stft = complex_stft.unsqueeze(1)

        # ================== Fullband ==================
        noisy_mag = rearrange(noisy_mag, "b f t -> b 1 f t")
        noisy_mag = noisy_mag**self.fdrc
        noisy_mag = noisy_mag[..., :-1, :]  # [B, 1, F, T]
        fb_input = noisy_mag[..., : self.fb_freqs, :]
        fb_input = self.norm(fb_input)
        fb_input = rearrange(fb_input, "b c f t -> b (c f) t")

        after_norm = time.time()
        print(f"encoder time: {(after_norm - start) / (16000 / 128) * 1000}ms")

        fb_output, fb_all_layer_outputs = self.fb_model(fb_input)  # [B, F, T]
        fb_output = rearrange(fb_output, "b f t -> b 1 f t")
        fb_output = fb_output.repeat(1, 1, self.num_repeats, 1)

        after_fb = time.time()
        print(f"(fb + linear) time: {(after_fb - after_norm) / (16000 / 128) * 1000}ms")

        # ================== Subband ==================
        # list [[B, df, F_1, T, 2], [B, df, F_2, T, 2], ...]
        df_coefs_list, sb_all_layer_outputs = self.sb_model(noisy_mag, fb_output)

        after_sb = time.time()

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
            window=torch.hann_window(self.win_length, device=noisy_y.device),
            length=noisy_y.size(-1),
        )

        after_istft = time.time()

        print(f"decoder time: {(after_istft - after_sb) / (16000 / 128) * 1000}ms")

        return enhanced_y, enhanced_mag, fb_all_layer_outputs, sb_all_layer_outputs


if __name__ == "__main__":
    import toml
    from torchinfo import summary

    from audiozen.metric import compute_synops

    config = toml.load(
        "/home/xianghao/proj/audiozen/recipes/intel_ndns/spike_fsb/baseline_s.toml"
        # "/home/xianghao/proj/audiozen/recipes/intel_ndns/spike_fsb/baseline_m_cumulative_laplace_norm.toml"
        # "/home/xianghao/proj/audiozen/recipes/intel_ndns/spike_fsb/baseline_l.toml"
    )
    model_args = config["model_g"]["args"]
    model_args.update({"bn": False})

    model = Separator(**model_args)
    input = torch.rand(1, 16000)
    y, mag, fb, sb = model(input)
    synops = compute_synops(fb, sb)
    neuron_ops = compute_neuronops(fb, sb)
    print(synops, neuron_ops)
