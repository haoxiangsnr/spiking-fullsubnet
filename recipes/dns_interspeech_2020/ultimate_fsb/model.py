import torch
import torch.nn as nn
import torchaudio as ta
from einops import rearrange

from audiozen.acoustics.audio_feature import stft
from audiozen.model.base_model import BaseModel
from audiozen.model.module.sequence_model import SequenceModel


class FSFBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fb = SequenceModel(
            input_size=32,
            output_size=32,
            hidden_size=256,
            num_layers=1,
            bidirectional=False,
            sequence_model="GRU",
            output_activate_function=None,
        )

        self.sb = SubbandModel(
            freq_cutoffs=[20, 80],
            sb_num_center_freqs=[1, 4, 8],
            sb_num_subband_freqs=[15, 15, 15],
            fb_num_center_freqs=[1, 4, 8],
            sequence_model="GRU",
            hidden_size=256,
            num_layers=1,
        )

        self.freq = SequenceModel(
            input_size=32,
            output_size=32,
            hidden_size=256,
            num_layers=1,
            bidirectional=False,
            sequence_model="GRU",
            output_activate_function=None,
        )

    def forward(self, noisy_spec):
        """
        Args:
            noisy_spec: [B, 2, F, T]
        """
        noisy_spec = noisy_spec[:, :, :32, :]
        fb_out = self.fb(noisy_spec)  # [B, 1, 32, T]
        fb_out = fb_out.repeat(1, 1, 8, 1)  # [B, 1, 256, T]

        sb_out = self.sb(noisy_spec, fb_out)  # [B, 2, 32, T]

        pass


class UltimateFullSubNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.fb_1 = SequenceModel(
            input_size=32,
            output_size=32,
            hidden_size=256,
            num_layers=1,
            bidirectional=False,
            sequence_model="GRU",
            output_activate_function=None,
        )

        self.fb_2 = SequenceModel(
            input_size=32,
            output_size=32,
            hidden_size=256,
            num_layers=1,
            bidirectional=False,
            sequence_model="GRU",
            output_activate_function=None,
        )

        self.sb_1 = SubbandModel(
            freq_cutoffs=[20, 80],
            sb_num_center_freqs=[1, 4, 8],
            sb_num_subband_freqs=[15, 15, 15],
            fb_num_center_freqs=[1, 4, 8],
            sequence_model="GRU",
            hidden_size=256,
            num_layers=1,
        )

        self.sb_2 = SubbandModel(
            freq_cutoffs=[20, 80],
            sb_num_center_freqs=[1, 4, 8],
            sb_num_subband_freqs=[15, 15, 15],
            fb_num_center_freqs=[1, 4, 8],
            sequence_model="GRU",
            hidden_size=256,
            num_layers=1,
        )

        self.freq_1 = SequenceModel(
            input_size=32,
            output_size=32,
            hidden_size=64,
            num_layers=1,
            bidirectional=False,
            sequence_model="GRU",
            output_activate_function=None,
        )

        self.freq_2 = SequenceModel(
            input_size=32,
            output_size=32,
            hidden_size=64,
            num_layers=1,
            bidirectional=False,
            sequence_model="GRU",
            output_activate_function=None,
        )

    def forward(self, noisy_y):
        noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy_y, 512, 256, 512)
        noisy_mag = noisy_mag**0.5

        noisy_mag_low = noisy_mag[:, :32, :]  # [B, 32, T]

        # fullband model 1
        fb_1_out = self.fb_1(noisy_mag_low)  # [B, 32, T]
        fb_1_out = rearrange(fb_1_out, "B F T -> B 1 F T")  # [B, 1, 32, T]

        # subband model 1
        sb_1_out = self.sb_1(noisy_mag, fb_1_out)  # [B, 2, 32, T]

        # frequency model 1
        freq_1_out = self.freq_1(sb_1_out)  # [B, 2, 32, T]

        # fullband model 2
        fb_2_out = self.fb_2(sb_1_out)  # [B, 2, 32, T]

        # subband model 2
        sb_2_out = self.sb_2(noisy_mag, fb_2_out)  # [B, 2, 32, T]

        # frequency model 2
        freq_2_out = self.freq_2(sb_2_out)  # [B, 2, 32, T]

        pass


class SubBandSequenceWrapper(SequenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, subband_input):
        (
            batch_size,
            num_subband_units,
            num_channels,
            num_subband_freqs,
            num_frames,
        ) = subband_input.shape
        assert num_channels == 1

        output = subband_input.reshape(
            batch_size * num_subband_units, num_subband_freqs, num_frames
        )
        output = super().forward(output)

        # [B, N, C, 2, center, T]
        output = output.reshape(batch_size, num_subband_units, 2, -1, num_frames)

        # [B, 2, N, center, T]
        output = output.permute(0, 2, 1, 3, 4).contiguous()

        # [B, C, N * F_subband_out, T]
        output = output.reshape(batch_size, 2, -1, num_frames)

        return output


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
        for (
            sb_num_center_freq,
            sb_num_neighbor_freq,
            fb_num_center_freq,
            fb_num_neighbor_freq,
        ) in zip(
            sb_num_center_freqs,
            sb_num_neighbor_freqs,
            fb_num_center_freqs,
            fb_num_neighbor_freqs,
        ):
            sb_models.append(
                SubBandSequenceWrapper(
                    input_size=(sb_num_center_freq + sb_num_neighbor_freq * 2)
                    + (fb_num_center_freq + fb_num_neighbor_freq * 2),
                    output_size=sb_num_center_freq * 2,
                    hidden_size=hidden_size,
                    num_layers=2,
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
        input: Tensor,
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
            input: complex spectrogram of shape (batch_size, 2, num_freqs, num_frames).
        """
        batch_size, num_channels, num_freqs, num_frames = noisy_input.size()

        subband_output = []
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
            subband_output.append(sb_model(sb_model_input))

        # [B, C, F, T]
        output = torch.cat(subband_output, dim=-2)

        return output
