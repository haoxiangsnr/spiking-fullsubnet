import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional

from audiozen.models.base_model import BaseModel
from audiozen.models.module.sequence_model import SequenceModel
from audiozen.models.module.tac import TransformAverageConcatenate

EPSILON = np.finfo(np.float32).eps


class SubbandSectionModel(BaseModel):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        sequence_model,
        output_activate_function,
        bidirectional,
    ):
        super().__init__()

        self.tac_layer_1 = TransformAverageConcatenate(input_size, input_size * 3)
        self.parallel_sequence_layer_1 = SequenceModel(
            input_size=input_size,
            output_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            sequence_model=sequence_model,
            output_activate_function=output_activate_function,
        )
        self.tac_layer_2 = TransformAverageConcatenate(input_size, input_size * 3)
        self.parallel_sequence_layer_2 = SequenceModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=1,
            bidirectional=bidirectional,
            sequence_model=sequence_model,
            output_activate_function=output_activate_function,
        )

    def forward(self, input):
        # [B, N, C, F_sb, T]
        batch_size, num_subband_units, num_channels, num_subband_freqs, _ = input.shape

        # TAC 1
        out = rearrange(input, "B N C F_sb T -> B N (C F_sb) T")
        out = self.tac_layer_1(out)
        out = rearrange(out, "B N (C F_sb) T -> (B N) (C F_sb) T", C=num_channels)

        # Parallel Subband 1
        out = self.parallel_sequence_layer_1(out)

        # TAC 2
        out = rearrange(
            out,
            "(B N) (C F_sb) T -> B N (C F_sb) T",
            B=batch_size,
            C=num_channels,
        )
        out = self.tac_layer_2(out)
        out = rearrange(out, "B N (C F_sb) T -> (B N) (C F_sb) T", C=num_channels)

        # Parallel Subband 2
        out = self.parallel_sequence_layer_2(out)
        out = rearrange(
            out,
            "(B N) (ri_dim F_ctr) T -> B ri_dim (N F_ctr) T",
            B=batch_size,
            ri_dim=2,
        )

        return out


class SubbandModel(BaseModel):
    def __init__(
        self,
        freq_cutoffs,
        sb_num_center_freqs,
        sb_num_neighbor_freqs,
        fb_num_center_freqs,
        fb_num_neighbor_freqs,
        sequence_model,
        hidden_size,
        activate_function=False,
        norm_type="offline_laplace_norm",
    ):
        super().__init__()

        sb_models = []
        for i in range(len(freq_cutoffs) - 1):
            sb_models.append(
                SubbandSectionModel(
                    input_size=(sb_num_center_freqs[i] + sb_num_neighbor_freqs[i] * 2)
                    + (fb_num_center_freqs[i] + fb_num_neighbor_freqs[i] * 2),
                    output_size=sb_num_center_freqs[i] * 2,
                    hidden_size=hidden_size,
                    sequence_model=sequence_model,
                    bidirectional=False,
                    output_activate_function=activate_function,
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
                f"The subband freqency interval is {upper_cutoff_freq-lower_cutoff_freq}."
            )

        # Extract valid input with the shape of [batch_size, 1, num_freqs, num_frames]
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
                (lower_cutoff_freq - num_neighbor_freqs) : (
                    upper_cutoff_freq + num_neighbor_freqs
                ),
                :,
            ]

        # unfold to [B, C * kernel_size, N]
        subband_unit_width = num_center_freqs + num_neighbor_freqs * 2
        output = functional.unfold(
            input=valid_input,
            kernel_size=(subband_unit_width, num_frames),
            stride=(num_center_freqs, num_frames),
        )

        output = rearrange(
            output,
            "B (C F_sb T) N -> B N C F_sb T",
            C=num_channels,
            F_sb=subband_unit_width,
        )

        return output

    def forward(self, noisy_input, fb_output):
        """Forward pass.

        Args:
            input: magnitude spectrogram of shape [B, C, F, T].
        """
        batch_size, num_channels, num_freqs, num_frames = noisy_input.size()
        assert num_channels == 1, "Only mono audio is supported."

        subband_output = []
        for sb_idx, sb_model in enumerate(self.sb_models):
            lower_cutoff_freq = self.freq_cutoffs[sb_idx]
            upper_cutoff_freq = self.freq_cutoffs[sb_idx + 1]

            # Unfold along frequency axis
            # [B, N, C, F_sb, T]
            noisy_subband = self._freq_unfold(
                noisy_input,
                lower_cutoff_freq=lower_cutoff_freq,
                upper_cutoff_freq=upper_cutoff_freq,
                num_center_freqs=self.sb_num_center_freqs[sb_idx],
                num_neighbor_freqs=self.sb_num_neighbor_freqs[sb_idx],
            )

            # [B, N, C, F_sb, T]
            fb_subband = self._freq_unfold(
                fb_output,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.fb_num_center_freqs[sb_idx],
                self.fb_num_neighbor_freqs[sb_idx],
            )

            sb_model_input = torch.cat([noisy_subband, fb_subband], dim=-2)
            sb_model_input = self.norm(sb_model_input)
            subband_output.append(sb_model(sb_model_input))

        # [B, C, F, T]
        output = torch.cat(subband_output, dim=-2)

        return output


class Separator(BaseModel):
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
        sequence_model,
        fb_output_activate_function,
        sb_output_activate_function,
        norm_type,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fdrc = fdrc

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
        )

        self.sb_model = SubbandModel(
            freq_cutoffs=freq_cutoffs,
            sb_num_center_freqs=sb_num_center_freqs,
            sb_num_neighbor_freqs=sb_num_neighbor_freqs,
            fb_num_center_freqs=fb_num_center_freqs,
            fb_num_neighbor_freqs=fb_num_neighbor_freqs,
            hidden_size=sb_hidden_size,
            sequence_model=sequence_model,
            activate_function=sb_output_activate_function,
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

        complex_stft_ri = torch.view_as_real(complex_stft)  # [B, F, T, 2]

        noisy_mag = torch.abs(complex_stft.unsqueeze(1))  # [B, 1, F, T]

        # ================== Fullband ==================
        noisy_mag = noisy_mag**self.fdrc  # fdrc
        noisy_mag = noisy_mag[..., :-1, :]  # [B, 1, F, T]
        fb_input = rearrange(self.norm(noisy_mag), "b c f t -> b (c f) t")
        fb_output = self.fb_model(fb_input)  # [B, F, T]
        fb_output = rearrange(fb_output, "b f t -> b 1 f t")

        # ================== Subband ==================
        cRM = self.sb_model(noisy_mag, fb_output)  # [B, 2, F, T]
        cRM = functional.pad(cRM, (0, 0, 0, 1), mode="constant", value=0.0)

        # ================== Masking ==================
        complex_stft_ri = rearrange(complex_stft_ri, "b f t c -> b c f t")
        enhanced_spec = cRM * complex_stft_ri  # [B, 2, F, T]

        # ================== ISTFT ==================
        enhanced_complex = torch.complex(
            enhanced_spec[:, 0, ...], enhanced_spec[:, 1, ...]
        )
        enhanced_y = torch.istft(
            enhanced_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            length=y.size(-1),
        )
        enhanced_y = enhanced_y.unsqueeze(1)  # [B, 1, T]
        return enhanced_y


if __name__ == "__main__":
    import toml
    from torchinfo import summary

    config = toml.load(
        # "recipes/dns_interspeech_2020/fsb/fix_distSampler_ctr124_4s_modelTD_1G.toml"
        # "recipes/dns_interspeech_2020/fsb/fix_distSampler_ctr124_4s_modelTD_hop128.toml"
        "recipes/dns_interspeech_2020/fsb/macs6G_subbandComm.toml"
    )

    model = Separator(**config["model"]["args"])
    # print(model)

    noisy_y = torch.rand(1, 16000)
    print(model(noisy_y).shape)
    summary(model, input_data=(noisy_y,), device="cpu")
