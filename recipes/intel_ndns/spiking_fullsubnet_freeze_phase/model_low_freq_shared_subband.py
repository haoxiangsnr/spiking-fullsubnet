from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel, SequenceModel
from einops import rearrange
from torch.nn import functional

from audiozen.acoustics.audio_feature import istft, stft
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
    complex_coefs = rearrange(complex_coefs, "b (c df) f t -> b c df f t", df=frame_size)

    # df
    out = torch.einsum("...ftn,...nft->...ft", complex_spec, complex_coefs)

    return out


class LinearWrapper(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass.

        Args:
            input: shape with [B, C, F, T] or [B, F, T]
        """
        if input.ndim == 4:
            batch_size, num_channels, num_freqs, num_frames = input.shape
            input = rearrange(input, "b c f t -> b t (c f)")

        output = super().forward(input)

        if input.ndim == 4:
            output = rearrange(output, "b t (c f) -> b c f t", c=num_channels)

        return output


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


class SubbandProcessor(BaseModel):
    def __init__(
        self,
        freq_cutoffs,
        sb_num_center_freqs,
        sb_num_neighbor_freqs,
        fb_num_center_freqs,
        fb_num_neighbor_freqs,
        sb_df_orders,
        bottleneck_size,
        sequence_model,
        hidden_size,
        activate_function=False,
        norm_type="offline_laplace_norm",
        shared_weights=False,
        bn=False,
    ):
        super().__init__()
        self.linear_in_layers = []
        self.linear_out_layers = []
        for sb_ctr_freq, sb_ngh_freq, fb_ctr_freq, fb_ngh_freq, sb_df_order in zip(
            sb_num_center_freqs,
            sb_num_neighbor_freqs,
            fb_num_center_freqs,
            fb_num_neighbor_freqs,
            sb_df_orders,
        ):
            self.linear_in_layers.append(
                LinearWrapper(
                    (sb_ctr_freq + sb_ngh_freq * 2) + (fb_ctr_freq + fb_ngh_freq * 2),
                    bottleneck_size,
                )
            )

            self.linear_out_layers.append(LinearWrapper(bottleneck_size, sb_ctr_freq * 2 * sb_df_order))

        self.linear_in_layers = nn.ModuleList(self.linear_in_layers)
        self.linear_out_layers = nn.ModuleList(self.linear_out_layers)

        # subband model
        self.sb_model = SubBandSequenceWrapper(
            input_size=bottleneck_size,
            output_size=bottleneck_size,
            hidden_size=hidden_size,
            num_layers=2,
            sequence_model=sequence_model,
            bidirectional=False,
            output_activate_function=activate_function,
            shared_weights=shared_weights,
            bn=bn,
        )

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
            valid_input = input[..., lower_cutoff_freq - num_neighbor_freqs : num_freqs, :]

            valid_input = functional.pad(
                input=valid_input,
                pad=(0, 0, 0, num_neighbor_freqs),
                mode="reflect",
            )
        else:
            # lower = lower_cutoff_freq - num_neighbor_freqs, upper = upper_cutoff_freq + num_neighbor_freqs
            valid_input = input[
                ...,
                lower_cutoff_freq - num_neighbor_freqs : upper_cutoff_freq + num_neighbor_freqs,
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
            noisy_input: magnitude spectrogram of shape (batch_size, 1, num_freqs, num_frames)
            fb_output: magnitude spectrogram of shape (batch_size, 1, num_freqs, num_frames)
        """
        batch_size, num_channels, num_freqs, num_frames = noisy_input.size()
        assert num_channels == 1, "Only mono audio is supported."

        subband_output = []
        subband_all_layer_outputs = []

        standard_sb_inputs = []
        lower_cutoff_freq = 0
        for ln_idx, ln_layer in enumerate(self.linear_in_layers):
            upper_cutoff_freq = lower_cutoff_freq + self.freq_cutoffs[ln_idx]

            # unfold frequency axis
            # [B, N, C, F_subband, T]
            noisy_subband = self._freq_unfold(
                noisy_input,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.sb_num_center_freqs[ln_idx],
                self.sb_num_neighbor_freqs[ln_idx],
            )

            # [B, N, C, F_subband, T]
            fb_subband = self._freq_unfold(
                fb_output,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.fb_num_center_freqs[ln_idx],
                self.fb_num_neighbor_freqs[ln_idx],
            )

            sb_input = torch.cat([noisy_subband, fb_subband], dim=-2)
            sb_input = rearrange(sb_input, "b n c f t -> b n (c f) t")
            sb_input = self.norm(sb_input)
            standard_sb_input = ln_layer(sb_input)
            standard_sb_input = rearrange(standard_sb_input, "b n (c f) t -> b n c f t", c=2)
            standard_sb_inputs.append(standard_sb_input)
            lower_cutoff_freq = upper_cutoff_freq

        # [B, N, C, F_subband, T]
        standard_sb_input = torch.cat(standard_sb_inputs, dim=-2)
        standard_sb_output, _ = self.sb_model(standard_sb_input)

        subband_output = []
        for ln_idx, ln_layer in enumerate(self.linear_out_layers):
            upper_cutoff_freq = lower_cutoff_freq + self.freq_cutoffs[ln_idx]

        # [B, C, F, T]
        # output = torch.cat(subband_output, dim=-2)
        return subband_output, subband_all_layer_outputs


class Model(BaseModel):
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

        self.sb_processor = SubbandProcessor(
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
        assert noisy_y.ndim in (2, 3), "Input must be 2D (B, T) or 3D tensor (B, 1, T)"
        if noisy_y.ndim == 2:
            noisy_y = noisy_y.unsqueeze(1)

        noisy_mag, _, noisy_real, noisy_imag = self.stft(noisy_y)
        noisy_stft = torch.complex(noisy_real, noisy_imag)

        # ================== Fullband ==================
        fb_input = noisy_mag**self.fdrc
        fb_input = fb_input[..., : self.fb_freqs, :]
        fb_input = self.norm(fb_input)
        fb_input = rearrange(fb_input, "b c f t -> b (c f) t")
        fb_output, fb_all_layer_outputs = self.fb_model(fb_input)  # [B, F, T]
        fb_output = rearrange(fb_output, "b f t -> b 1 f t")
        fb_output = fb_output.repeat(1, 1, self.num_repeats, 1)

        # ================== Subband ===================
        # list [[B, df, F_1, T, 2], [B, df, F_2, T, 2], ...]
        df_coefs_list, sb_all_layer_outputs = self.sb_processor(noisy_mag, fb_output)

        # ================== Iterative Masking ==================
        num_filtered = 0
        enhanced_spec_list = []
        for df_coefs, df_order in zip(df_coefs_list, self.sb_df_orders):
            # [B, C, F , T] = [B, C, F, ]
            num_sub_freqs = df_coefs.shape[2]
            complex_stft_in = noisy_stft[..., num_filtered : num_filtered + num_sub_freqs, :]
            enhanced_subband = deepfiltering(complex_stft_in, df_coefs, frame_size=df_order)  # [B, 1, F, T] of complex
            enhanced_spec_list.append(enhanced_subband)
            num_filtered += num_sub_freqs

        # [B, C, F, T]
        enhanced_spec = torch.cat(enhanced_spec_list, dim=-2)

        enhanced_stft = noisy_stft.clone()  # [B, C, F， T]

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
        return enhanced_y, enhanced_mag, fb_all_layer_outputs, sb_all_layer_outputs


if __name__ == "__main__":
    import toml

    from audiozen.metric import compute_synops

    config = toml.load(
        # "/home/xianghao/proj/audiozen/recipes/intel_ndns/spike_fsb/baseline_s.toml"
        "/home/xianghao/proj/audiozen/recipes/intel_ndns/spike_fsb/baseline_m.toml"
        # "/home/xianghao/proj/audiozen/recipes/intel_ndns/spike_fsb/baseline_l.toml"
    )
    model_args = config["model_g"]["args"]

    model = Model(**model_args)
    input = torch.rand(5, 160000)
    y, mag, fb, sb = model(input)
    synops = compute_synops(fb, sb)
    neuron_ops = compute_neuronops(fb, sb)
    print(synops, neuron_ops)
