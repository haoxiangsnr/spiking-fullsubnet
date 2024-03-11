from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.models.spiking_fullsubnet.efficient_spiking_neuron import MemoryState, efficient_spiking_neuron


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


class SubBandSequenceModel(SequenceModel):
    def __init__(self, df_order, num_spks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_order = df_order
        self.num_spks = num_spks

    def forward(self, input_features):
        """How to process the subband features.

        - `fs`: number of frequency bins in the subband.
        - `fc`: number of frequency bins in the center frequency.
        - `df`: order of the deep filter.
        - `n`: number of subbands.
        - `s`: number of speakers.
        - `c`: number of channels.

        Args:
            input (`torch.Tensor` of shape `(batch_size, num_subbands, num_channels, sb_freq_size, sequence_length)`):
                Subband feature at a particular frequency.

        Returns:
            output (`torch.Tensor` of shape `(batch_size, df_order, num_subbands * ctr_freq_size, sequence_length, num_channels)`):
                Complex output tensor. The last dimension (`num_channels`) is the real and imaginary parts.
        """
        batch_size, num_subbands, num_channels, sb_freq_size, sequence_length = input_features.shape
        assert num_channels == 1, "Only mono audio is supported."

        input_features = rearrange(input_features, "b n c fs t -> (b n) (c fs) t")

        output, all_layer_outputs = super().forward(input_features)

        # `2 * num_channels` because we have real and imaginary parts in the output.
        output = rearrange(
            output,
            "(b n) (c fc df s) t -> b df s (n fc) t c",
            b=batch_size,
            s=self.num_spks,
            c=num_channels * 2,
            df=self.df_order,
        )

        return output, all_layer_outputs


class SubbandModel(nn.Module):
    def __init__(
        self,
        freq_cutoffs,
        center_freq_sizes,
        neighbor_freq_sizes,
        df_orders,
        num_spks,
        **kwargs,
    ):
        """Subband model.

        Args:
            freq_cutoffs (`list` of `int`):
                 Cutoff frequencies for the subbands. The first and last elements are the lower and upper cutoffs.
            center_freq_sizes (`list` of `int`):
                Number of frequency bins in the center frequency for each subband. The length of this list must be
                equal to the length of `freq_cutoffs` minus 1.
            neighbor_freq_sizes: like `sb_center_freq_sizes`, but for the neighboring frequency bins.
            df_orders: like `center_freq_sizes`, but for the deep filter order.
            num_spks (`int`): Number of speakers.
            kwargs: other arguments for `SequenceModel`.
        """
        super().__init__()
        assert len(freq_cutoffs) - 1 == len(center_freq_sizes), "Number of subbands must be equal to len(cutoffs)."

        sb_models = []
        for ctr_freq, nbr_freq, df_order in zip(center_freq_sizes, neighbor_freq_sizes, df_orders):
            sb_models.append(
                SubBandSequenceModel(
                    input_size=(ctr_freq + nbr_freq * 2) + ctr_freq,
                    proj_size=2 * ctr_freq * df_order * num_spks,
                    df_order=df_order,
                    num_spks=num_spks,
                    **kwargs,
                )
            )

        self.sb_models = nn.ModuleList(sb_models)
        self.freq_cutoffs = freq_cutoffs
        self.center_freq_sizes = center_freq_sizes
        self.neighbor_freq_sizes = neighbor_freq_sizes
        self.df_orders = df_orders

    def forward(self, noisy_input, fb_output):
        """Frequency-wise processing of the subband features.

        Separate the noisy input into several sections. Each section has the same center frequency and neighboring
        frequency bins. Then, process each section with the corresponding subband model.

        For a subband feature at frequency `f`, the corresponding feature is obtained by concatenating:
        1. `f` itself.
        2. `f - N` to `f - 1`, where `N` is the number of frequency bins on each side of `f`.
        3. `f + 1` to `f + N`.
        4. corresponding frequency bins in the fullband feature.

        Args:
            noisy_input (`torch.Tensor` of shape `(batch_size, num_channels, num_freqs, num_frames)`):
                Noisy input spectrogram. `num_channels` must be 1.
            fb_output (`torch.Tensor` of shape `(batch_size, num_channels, num_spks, num_freqs, num_frames)`):
                Fullband output spectrogram.
        """
        batch_size, num_channels, num_freqs, num_frames = noisy_input.size()
        assert num_channels == 1, "Only mono audio is supported."

        output = []
        all_layer_outputs = []
        for idx, sb_model in enumerate(self.sb_models):
            # [batch_size, num_subbands, num_channels, sb_freq_size, num_frames]
            noisy_subbands = self._freq_unfold(
                input=noisy_input,
                lower_cutoff_freq=self.freq_cutoffs[idx],
                upper_cutoff_freq=self.freq_cutoffs[idx + 1],
                ctr_freq=self.center_freq_sizes[idx],
                nbr_freq=self.neighbor_freq_sizes[idx],
            )

            fb_subbands = self._freq_unfold(
                input=fb_output,
                lower_cutoff_freq=self.freq_cutoffs[idx],
                upper_cutoff_freq=self.freq_cutoffs[idx + 1],
                ctr_freq=self.center_freq_sizes[idx],
                nbr_freq=0,
            )

            # Concatenate the subband features with the corresponding fullband features.
            sb_input = torch.cat([noisy_subbands, fb_subbands], dim=-2)
            sb_output, sb_all_layer_outputs = sb_model(sb_input)
            output += [sb_output]
            all_layer_outputs += [sb_all_layer_outputs]

        return output, all_layer_outputs

    def _freq_unfold(self, input, lower_cutoff_freq, upper_cutoff_freq, ctr_freq, nbr_freq):
        """Unfold the frequency bins based on a given lower and upper cutoff frequency bondaries.

        Args:
            input (`torch.Tensor` of shape `(batch_size, num_channels, num_freqs, num_frames)`):
                Noisy input spectrogram.
            lower_cutoff_freq: lower cutoff frequency of current section.
            upper_cutoff_freq: upper cutoff frequency of current section.
            ctr_freq: number of frequency bins in the center frequency.
            nbr_freq: number of neighboring frequency bins.

        Returns:
            output (`torch.Tensor` of shape `(batch_size, num_subbands, num_channels, sb_freq_size, num_frames)`):
                Unfolded tensor.
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        assert num_channels == 1, "Only mono audio is supported."

        if (upper_cutoff_freq - lower_cutoff_freq) % ctr_freq != 0:
            raise ValueError(
                f"Number of frequency bins must be divisible by the center frequency."
                f"GOT: {ctr_freq=}, {upper_cutoff_freq=}, {lower_cutoff_freq=}"
            )

        # extract valid input with the shape of [batch_size, 1, num_subbands * ctr_freq + 2 * nbr_freq, num_frames]
        if lower_cutoff_freq == 0:
            # lower_cutoff_freq = 0 is a special case.
            # lower = 0, upper = upper_cutoff_freq + nbr_freq
            valid_input = input[..., : upper_cutoff_freq + nbr_freq, :]
            valid_input = F.pad(valid_input, (0, 0, nbr_freq, 0), mode="reflect")
        elif upper_cutoff_freq == num_freqs:
            # upper_cutoff_freq = num_freqs is a special case.
            # lower = lower_cutoff_freq - nbr_freq, upper = num_freqs
            valid_input = input[..., lower_cutoff_freq - nbr_freq :, :]
            valid_input = F.pad(valid_input, (0, 0, 0, nbr_freq), mode="reflect")
        else:
            # lower = lower_cutoff_freq - nbr_freq, upper = upper_cutoff_freq + nbr_freq
            valid_input = input[..., lower_cutoff_freq - nbr_freq : upper_cutoff_freq + nbr_freq, :]

        # Unfold the frequency bins.
        output = F.unfold(
            input=valid_input, kernel_size=(ctr_freq + nbr_freq * 2, num_frames), stride=(ctr_freq, num_frames)
        )

        # Reshape the output to [batch_size, num_subbands, num_channels, sb_freq_size, num_frames].
        output = rearrange(output, "b (c fs t) n -> b n c fs t", c=num_channels, fs=ctr_freq + nbr_freq * 2)

        return output


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


class SpikingFullSubNet(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        fdrc,
        fb_input_size,
        fb_hidden_size,
        fb_num_layers,
        fb_proj_size,
        fb_output_activate_function,
        sb_hidden_size,
        sb_num_layers,
        freq_cutoffs,
        df_orders,
        center_freq_sizes,
        neighbor_freq_sizes,
        use_pre_layer_norm_fb=True,
        use_pre_layer_norm_sb=True,
        bn=False,
        shared_weights=False,
        sequence_model="GSN",
        num_spks=1,
    ):
        super().__init__()

        self.fb_model = SequenceModel(
            input_size=fb_input_size,
            hidden_size=fb_hidden_size,
            num_layers=fb_num_layers,
            shared_weights=shared_weights,
            sequence_model=sequence_model,
            proj_size=fb_proj_size,
            output_activate_function=fb_output_activate_function,
            bn=bn,
            use_pre_layer_norm=use_pre_layer_norm_fb,
        )

        self.sb_model = SubbandModel(
            freq_cutoffs=freq_cutoffs,
            center_freq_sizes=center_freq_sizes,
            neighbor_freq_sizes=neighbor_freq_sizes,
            df_orders=df_orders,
            num_spks=num_spks,
            hidden_size=sb_hidden_size,
            num_layers=sb_num_layers,
            shared_weights=shared_weights,
            sequence_model=sequence_model,
            bn=bn,
            use_pre_layer_norm=use_pre_layer_norm_sb,
        )

        self.subband_model = None

        self.stft = partial(stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.istft = partial(istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        self.fb_input_size = fb_input_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fdrc = fdrc
        self.df_orders = df_orders
        self.num_spks = num_spks

    def forward(self, input):
        """Forward function.

        Args:
            input (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Waveform tensor.

        Returns:
            output (`torch.Tensor` of shape `(batch_size, sequence_length) or `(batch_size, num_spks, sequence_length)`):
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
        noisy_mag = noisy_mag[..., :-1, :]

        # Extract the lowband signal.
        fb_input = noisy_mag[..., : self.fb_input_size, :]
        fb_input = rearrange(fb_input, "b c f t -> b (c f) t")
        fb_output, fb_all_layer_outputs = self.fb_model(fb_input)
        fb_output = rearrange(fb_output, "b f t -> b 1 f t")
        fb_output = fb_output.repeat(1, 1, (self.n_fft // 2 + 1) // self.fb_input_size, 1)

        # ================== Subband model ==================
        # list [[B, df, S, F_1, T, 2], [B, df, S, F_2, T, 2], ...], where F = F_1 + F_2 + ...
        df_coef_list, sb_all_layer_outputs = self.sb_model(noisy_mag, fb_output)

        # ================== Reconstruct the output ==================
        num_filtered_freqs = 0
        enh_freqs_list = []
        for df_coef, df_order in zip(df_coef_list, self.df_orders):
            num_freqs = df_coef.shape[3]
            # [B, c, f, t]
            comp_stft_in = noisy_cmp[..., num_filtered_freqs : num_filtered_freqs + num_freqs, :]
            enh_freqs = deepfiltering(comp_stft_in, df_coef, df_order, self.num_spks)  # [B, c, s, f, t]
            enh_freqs_list.append(enh_freqs)
            num_filtered_freqs += num_freqs

        enh_freqs = torch.cat(enh_freqs_list, dim=-2)  # [B, c, s, F, T]、
        enh_stft = repeat(noisy_cmp, "b 1 f t -> b 1 s f t", s=self.num_spks).clone()

        if self.num_spks > 1:
            enh_stft[..., :-1, :] = enh_freqs
            enh_stft = rearrange(enh_stft, "b 1 s f t -> (b s) f t")
            enh_y = self.istft(enh_stft, length=sequence_length)
            enh_y = rearrange(enh_y, "(b s) t -> b s t", s=self.num_spks)
            return enh_y, fb_all_layer_outputs, sb_all_layer_outputs
        else:
            enh_stft[..., :-1, :] = enh_freqs
            enh_stft = rearrange(enh_stft, "b 1 1 f t -> b f t")
            enh_mag = torch.abs(enh_stft)  # For computing DNSMOS loss
            enh_y = self.istft(enh_stft, length=sequence_length)
            return enh_y, enh_mag, fb_all_layer_outputs, sb_all_layer_outputs


if __name__ == "__main__":
    from torchinfo import summary

    from audiozen.metric import compute_neuronops, compute_synops, compute_synops_v2

    model = SpikingFullSubNet(
        n_fft=512,
        hop_length=128,
        win_length=512,
        fdrc=0.5,
        fb_input_size=64,
        fb_hidden_size=240,
        fb_num_layers=2,
        fb_proj_size=64,
        fb_output_activate_function=None,
        sb_hidden_size=160,
        sb_num_layers=2,
        freq_cutoffs=[0, 32, 128, 256],
        df_orders=[3, 1, 1],
        center_freq_sizes=[8, 32, 64],
        neighbor_freq_sizes=[15, 15, 15],
        use_pre_layer_norm_fb=True,
        use_pre_layer_norm_sb=True,
        bn=True,
        shared_weights=True,
        sequence_model="GSN",
        num_spks=1,
    )

    input = torch.rand(2, 16000)
    summary(model, input_data=input)
    *_, fb_all_layer_outputs, sb_all_layer_outputs = model(input)

    synops = compute_synops(fb_all_layer_outputs, sb_all_layer_outputs, shared_weights=True)
    synops_v2 = compute_synops_v2(fb_all_layer_outputs, sb_all_layer_outputs, shared_weights=True)
    neuronops = compute_neuronops(fb_all_layer_outputs, sb_all_layer_outputs)

    buffer_latency = 0.032
    enc_dec_latency = 0.030 / 1000
    dns_latency = 0
    dt = 0.008

    latency = buffer_latency + enc_dec_latency + dns_latency
    effective_synops_rate = (synops + 10 * neuronops) / dt
    effective_synops_rate_v2 = (synops_v2 + 10 * neuronops) / dt
    synops_delay_product = effective_synops_rate * latency
    synops_delay_product_v2 = effective_synops_rate_v2 * latency

    print(f"synops: {synops}, neuronops: {neuronops}, synops_v2: {synops_v2}")
    print(synops + 10 * neuronops)
    print(f"Solution Latency                 : {latency * 1000: .3f} ms")
    print(f"Power proxy (Effective SynOPS)   : {effective_synops_rate:.3f} ops/s")
    print(f"PDP proxy (SynOPS-delay product) : {synops_delay_product: .3f} ops")

    print(f"Power proxy v2 (Effective SynOPS)   : {effective_synops_rate_v2 / 1000000 * 0.93 :.3f} ops/s")
    print(f"PDP proxy v2 (SynOPS-delay product) : {synops_delay_product_v2 / 1000000 * 0.93 : .3f} ops")
