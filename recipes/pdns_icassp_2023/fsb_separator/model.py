from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.model.base_model import BaseModel
from audiozen.model.module.sequence_model import SequenceModel


class AdaptLayer(nn.Module):
    def __init__(self, main_dim, enroll_dim):
        super().__init__()
        self.adaptor = nn.Linear(main_dim + enroll_dim, main_dim)

    def forward(self, main, enroll):
        # main: [B, C, F, T]
        # enroll: [B, C, F]
        input = torch.cat((main, enroll[..., None].broadcast_to(main.shape)), dim=-2)
        input = rearrange(input, "b c f t -> b t (c f)")
        out = self.adaptor(input)
        out = rearrange(out, "b t (c f) -> b c f t", c=main.shape[1], f=main.shape[2])
        return out


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
        adaptor_num_center_freqs,
        adaptor_num_neighbor_freqs,
        sequence_model,
        hidden_size,
        dropout,
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
            adaptor_num_center_freq,
            adaptor_num_neighbor_freq,
        ) in zip(
            sb_num_center_freqs,
            sb_num_neighbor_freqs,
            fb_num_center_freqs,
            fb_num_neighbor_freqs,
            adaptor_num_center_freqs,
            adaptor_num_neighbor_freqs,
        ):
            sb_models.append(
                SubBandSequenceWrapper(
                    input_size=(sb_num_center_freq + sb_num_neighbor_freq * 2)
                    + (fb_num_center_freq + fb_num_neighbor_freq * 2)
                    + (adaptor_num_center_freq + adaptor_num_neighbor_freq * 2),
                    output_size=sb_num_center_freq * 2,
                    hidden_size=hidden_size,
                    num_layers=2,
                    sequence_model=sequence_model,
                    bidirectional=False,
                    output_activate_function=activate_function,
                    dropout=dropout,
                )
            )

        self.sb_models = nn.ModuleList(sb_models)
        self.freq_cutoffs = freq_cutoffs
        self.sb_num_center_freqs = sb_num_center_freqs
        self.sb_num_neighbor_freqs = sb_num_neighbor_freqs
        self.fb_num_center_freqs = fb_num_center_freqs
        self.fb_num_neighbor_freqs = fb_num_neighbor_freqs
        self.adaptor_num_center_freqs = adaptor_num_center_freqs
        self.adaptor_num_neighbor_freqs = adaptor_num_neighbor_freqs

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

    def forward(self, noisy_input, fb_output, adaptor_output):
        """Forward pass.

        Args:
            input: magnitude spectrogram of shape (batch_size, 1, num_freqs, num_frames).
        """
        batch_size, num_channels, num_freqs, num_frames = noisy_input.size()
        assert num_channels == 1, "Only mono audio is supported."

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

            # [B, N, C, F_subband, T]
            adaptor_subband = self._freq_unfold(
                adaptor_output,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.adaptor_num_center_freqs[sb_idx],
                self.adaptor_num_neighbor_freqs[sb_idx],
            )

            sb_model_input = torch.cat(
                [noisy_subband, fb_subband, adaptor_subband], dim=-2
            )
            sb_model_input = self.norm(sb_model_input)
            subband_output.append(sb_model(sb_model_input))

        # [B, C, F, T]
        output = torch.cat(subband_output, dim=-2)

        return output


class Model(BaseModel):
    def __init__(
        self,
        fdrc,
        n_fft,
        hop_length,
        win_length,
        fb_freqs,
        num_freqs,
        sequence_model,
        fb_hidden_size,
        adapt_fb_hidden_size,
        adapt_embedding_hidden_size,
        fb_output_activate_function,
        freq_cutoffs,
        sb_num_center_freqs,
        sb_num_neighbor_freqs,
        fb_num_center_freqs,
        fb_num_neighbor_freqs,
        adaptor_num_center_freqs,
        adaptor_num_neighbor_freqs,
        sb_hidden_size,
        sb_output_activate_function,
        dropout,
        norm_type,
    ):
        """FullSubNet with explicitly separated subbands.

        Args:
            fdrc: pow factor of feature dynamic range compression (fdrc).
            n_fft: number of fft size.
            hop_length: hop size of stft.
            win_length: window size of stft.
            fb_freqs: number of low frequency bins extracted from fullband model input.
            num_freqs: number of frequency bins.
            sequence_model: which sequence model to use.
            fb_hidden_size: hidden size of the fullband model.
            fb_output_activate_function: output activation function of the fullband model.
            freq_cutoffs: _description_
            sb_num_center_freqs: _description_
            sb_num_neighbor_freqs: _description_
            fb_num_center_freqs: _description_
            fb_num_neighbor_freqs: _description_
            sb_hidden_size: _description_
            sb_output_activate_function: _description_
            norm_type: _description_
        """
        super().__init__()
        assert (
            num_freqs % fb_freqs
        ) == 0, "The fb_freqs should be divisible by the num_freqs."

        self.fdrc = fdrc
        self.fb_freqs = fb_freqs
        self.repeat_num = num_freqs // fb_freqs

        self.fb_model = SequenceModel(
            input_size=fb_freqs,
            output_size=fb_freqs,
            hidden_size=fb_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
            dropout=dropout,
        )

        self.adapt_fb_encoder = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=adapt_fb_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
            dropout=dropout,
        )

        self.adapt_embedding_encoder = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=adapt_embedding_hidden_size,
            num_layers=2,
            bidirectional=True,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
            dropout=dropout,
        )

        self.adapt_layer = AdaptLayer(
            main_dim=num_freqs,
            enroll_dim=num_freqs,
        )

        self.sb_model = SubbandModel(
            freq_cutoffs=freq_cutoffs,
            sb_num_center_freqs=sb_num_center_freqs,
            sb_num_neighbor_freqs=sb_num_neighbor_freqs,
            fb_num_center_freqs=fb_num_center_freqs,
            fb_num_neighbor_freqs=fb_num_neighbor_freqs,
            adaptor_num_center_freqs=adaptor_num_center_freqs,
            adaptor_num_neighbor_freqs=adaptor_num_neighbor_freqs,
            hidden_size=sb_hidden_size,
            sequence_model=sequence_model,
            activate_function=sb_output_activate_function,
            dropout=dropout,
        )

        self.norm = self.norm_wrapper(norm_type)

        self.stft = partial(
            stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.istft = partial(
            istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )

    def forward(self, noisy_y, enrollment_y):
        noisy_mag, _, noisy_real, noisy_imag = self.stft(noisy_y)
        noisy_mag = noisy_mag**self.fdrc  # fdrc
        noisy_mag = noisy_mag.unsqueeze(1)  # [B, 1, F, T]
        noisy_mag = noisy_mag[..., :-1, :]  # [B, 1, F, T]
        num_frames = noisy_mag.size(-1)

        enrollment_mag, _, enrollment_real, enrollment_imag = self.stft(enrollment_y)
        enrollment_mag = enrollment_mag**self.fdrc  # fdrc
        enrollment_mag = enrollment_mag.unsqueeze(1)  # [B, 1, F, T]
        enrollment_mag = enrollment_mag[..., :-1, :]  # [B, 1, F, T]
        batch_size, num_channels, emb_size, num_enroll_frames = enrollment_mag.size()

        # ===================================
        # Fullband model (use Lower part of the frequency for global information)
        # ===================================
        fb_input = noisy_mag[..., : self.fb_freqs, :]  # [B, 1, self.fb_freqs, T]
        fb_input = self.norm(fb_input)
        fb_input = rearrange(fb_input, "b c f t -> b (c f) t")
        fb_output = self.fb_model(fb_input)
        fb_output = rearrange(fb_output, "b (c f) t -> b c f t", c=num_channels)
        fb_output = fb_output.repeat(1, 1, self.repeat_num, 1)

        # ===================================
        # Adaptive filter
        # ===================================
        # fullband encoder
        adapt_fb_input = self.norm(noisy_mag)
        adapt_fb_input = rearrange(adapt_fb_input, "b c f t -> b (c f) t")
        adaptor_fb_output = self.adapt_fb_encoder(adapt_fb_input)
        adaptor_fb_output = rearrange(
            adaptor_fb_output, "b (c f) t -> b c f t", c=num_channels
        )
        # embedding encoder
        enrollment_mag = self.norm(enrollment_mag)
        enrollment_mag = rearrange(enrollment_mag, "b c f t -> b (c f) t")
        embedding = self.adapt_embedding_encoder(enrollment_mag)
        embedding = rearrange(embedding, "b (c f) t -> b c f t", c=num_channels)
        embedding = embedding.mean(dim=3)  # [B, 1, F]
        # adapt layer
        adaptor_output = self.adapt_layer(adaptor_fb_output, embedding)  # [B, 1, F, T]

        # ===================================
        # Subband model
        # ===================================
        cRM = self.sb_model(noisy_mag, fb_output, adaptor_output)

        # ===================================
        # Return enhanced signal
        # ===================================
        cRM = functional.pad(cRM, (0, 0, 0, 1), mode="constant", value=0.0)
        enhanced_real = cRM[:, 0, ...] * noisy_real
        enhanced_imag = cRM[:, 1, ...] * noisy_imag

        return enhanced_real, enhanced_imag


if __name__ == "__main__":
    import time

    from torchinfo import summary

    with torch.no_grad():
        model = Model(
            fdrc=0.5,
            n_fft=320,
            hop_length=160,
            win_length=320,
            fb_freqs=80,
            num_freqs=160,
            sequence_model="LSTM",
            fb_hidden_size=768,
            adapt_fb_hidden_size=512,
            adapt_embedding_hidden_size=512,
            fb_output_activate_function=False,
            freq_cutoffs=[20, 80],
            sb_num_center_freqs=[1, 4, 8],
            sb_num_neighbor_freqs=[15, 15, 15],
            fb_num_center_freqs=[1, 4, 8],
            fb_num_neighbor_freqs=[0, 0, 0],
            adaptor_num_center_freqs=[1, 4, 8],
            adaptor_num_neighbor_freqs=[2, 2, 2],
            sb_hidden_size=384,
            dropout=0.1,
            sb_output_activate_function=False,
            norm_type="offline_laplace_norm",
        )
        noisy_y = torch.rand(1, 16000)
        embedding = torch.rand(1, 16000)
        start = time.time()
        enhanced_real, enhanced_imag = model(noisy_y, embedding)
        print(time.time() - start)
