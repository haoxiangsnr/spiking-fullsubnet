import torch
import torch.nn as nn
from einops import pack, rearrange, unpack
from torch.nn import functional

from audiozen.constant import EPSILON
from audiozen.models.base_model import BaseModel
from audiozen.models.module.res_rnn import GroupMLP
from audiozen.models.module.sequence_model import SequenceModel
from audiozen.models.module.tac import TransformAverageConcatenate


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
        sb_ctr_freqs,
        sb_ngb_freqs,
        fb_ctr_freqs,
        fb_ngb_freqs,
        sequence_model,
        hidden_size,
        freq_comm_hidden_size,
        num_groups,
        interlayer_feat_dim,
        activate_function=False,
        norm_type="offline_laplace_norm",
    ):
        super().__init__()
        input_reshaping_layers = []
        sb_models = []
        output_reshaping_layers = []
        for i in range(len(freq_cutoffs) - 1):
            input_size = (
                (sb_ctr_freqs[i] + sb_ngb_freqs[i] * 2)
                + (fb_ctr_freqs[i] + fb_ngb_freqs[i] * 2)
            ) * 2

            input_reshaping_layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(input_size, eps=EPSILON),
                    nn.Conv1d(input_size, interlayer_feat_dim, 1),
                )
            )

            sb_models.append(
                SequenceModel(
                    input_size=interlayer_feat_dim,
                    hidden_size=hidden_size,
                    output_size=interlayer_feat_dim,
                    num_layers=1,
                    bidirectional=False,
                    sequence_model=sequence_model,
                    output_activate_function=activate_function,
                )
            )

            output_reshaping_layers.append(
                GroupMLP(
                    input_size=interlayer_feat_dim,
                    hidden_size=hidden_size,
                    output_size=sb_ctr_freqs[i] * 2,
                    num_groups=num_groups,
                )
            )

        self.freq_comm = SequenceModel(
            input_size=interlayer_feat_dim,
            hidden_size=freq_comm_hidden_size,
            output_size=interlayer_feat_dim,
            num_layers=1,
            bidirectional=True,
            sequence_model=sequence_model,
            output_activate_function=activate_function,
        )

        self.input_reshaping_layers = nn.ModuleList(input_reshaping_layers)
        self.sb_models = nn.ModuleList(sb_models)
        self.output_reshaping_layers = nn.ModuleList(output_reshaping_layers)

        self.freq_cutoffs = freq_cutoffs
        self.sb_num_center_freqs = sb_ctr_freqs
        self.sb_num_neighbor_freqs = sb_ngb_freqs
        self.fb_num_center_freqs = fb_ctr_freqs
        self.fb_num_neighbor_freqs = fb_ngb_freqs

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

        section_out = []
        for i in range(len(self.freq_cutoffs) - 1):
            lower_cutoff_freq = self.freq_cutoffs[i]
            upper_cutoff_freq = self.freq_cutoffs[i + 1]

            noisy_subband = self._freq_unfold(
                noisy_input,
                lower_cutoff_freq=lower_cutoff_freq,
                upper_cutoff_freq=upper_cutoff_freq,
                num_center_freqs=self.sb_num_center_freqs[i],
                num_neighbor_freqs=self.sb_num_neighbor_freqs[i],
            )  # [B, N, C, F_sb, T]

            fb_subband = self._freq_unfold(
                fb_output,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.fb_num_center_freqs[i],
                self.fb_num_neighbor_freqs[i],
            )

            input, _ = pack([noisy_subband, fb_subband], "B N C * T")
            input = rearrange(input, "B N C F_sb T -> (B N) (C F_sb) T")
            out = self.input_reshaping_layers[i](input)
            out = self.sb_models[i](out)  # [B * N, H, T]

            out = rearrange(out, "(B N) H T -> B N H T", B=batch_size)
            section_out.append(out)

        section_out, ps = pack(section_out, "B * H T")
        section_out = rearrange(section_out, "B N H T -> (B T) H N")
        comm_out = self.freq_comm(section_out)
        comm_out = rearrange(comm_out, "(B T) H N -> B N H T", B=batch_size)

        comm_out = unpack(comm_out, ps, "B * H T")

        mask_output = []
        for i in range(len(self.freq_cutoffs) - 1):
            input = rearrange(comm_out[i], "B N H T -> (B N) H T")
            out = self.output_reshaping_layers[i](input)
            out = rearrange(
                out,
                "(B N) (ri_dim H) T -> B ri_dim (N H) T",
                B=batch_size,
                ri_dim=2,
            )
            mask_output.append(out)

        mask_output, _ = pack(mask_output, "B ri_dim * T")

        # [B, C, F, T]
        return mask_output


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
        freq_comm_hidden_size,
        interlayer_feat_dim,
        num_groups,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fdrc = fdrc

        self.fb_model = SequenceModel(
            input_size=num_freqs * 2,
            output_size=num_freqs * 2,
            hidden_size=fb_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
        )

        self.subband_models = [
            SubbandModel(
                freq_cutoffs=freq_cutoffs,
                sb_ctr_freqs=sb_num_center_freqs,
                sb_ngb_freqs=sb_num_neighbor_freqs,
                fb_ctr_freqs=fb_num_center_freqs,
                fb_ngb_freqs=fb_num_neighbor_freqs,
                hidden_size=sb_hidden_size,
                freq_comm_hidden_size=freq_comm_hidden_size,
                interlayer_feat_dim=interlayer_feat_dim,
                num_groups=num_groups,
                sequence_model=sequence_model,
                activate_function=sb_output_activate_function,
            ),
            SubbandModel(
                freq_cutoffs=freq_cutoffs,
                sb_ctr_freqs=sb_num_center_freqs,
                sb_ngb_freqs=sb_num_neighbor_freqs,
                fb_ctr_freqs=fb_num_center_freqs,
                fb_ngb_freqs=fb_num_neighbor_freqs,
                hidden_size=sb_hidden_size,
                freq_comm_hidden_size=freq_comm_hidden_size,
                interlayer_feat_dim=interlayer_feat_dim,
                num_groups=num_groups,
                sequence_model=sequence_model,
                activate_function=sb_output_activate_function,
            ),
        ]

        self.norm = self.norm_wrapper(norm_type)
        self.subband_models = nn.ModuleList(self.subband_models)

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
        complex_stft_ri = rearrange(complex_stft_ri, "B F T C -> B C F T")

        batch_size, num_channels, num_freqs, num_frames = complex_stft_ri.shape

        # ================== Fullband ==================
        fb_input = complex_stft_ri**self.fdrc  # fdrc
        fb_input = fb_input[..., :-1, :]  # [B, 1, F, T]
        fb_output = rearrange(self.norm(fb_input), "B C F T -> B (C F) T")
        fb_output = self.fb_model(fb_output)  # [B, F, T]
        fb_output = rearrange(fb_output, "B (C F) T -> B C F T", C=num_channels)

        # ================== Subband ==================
        output = fb_output
        for i, subband_model in enumerate(self.subband_models):
            output = subband_model(fb_input, output)

        # ================== Masking ==================
        cRM = functional.pad(output, (0, 0, 0, 1), mode="constant", value=0.0)
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
        "recipes/dns_interspeech_2020/fsb/macs6G_globalCommInterLayer32.toml"
    )

    model = Separator(**config["model"]["args"])
    # print(model)

    noisy_y = torch.rand(1, 16000)
    print(model(noisy_y).shape)
    summary(model, input_data=(noisy_y,), device="cpu")
