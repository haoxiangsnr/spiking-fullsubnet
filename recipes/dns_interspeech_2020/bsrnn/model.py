import numpy as np
import torch
import torch.nn as nn


class TAC(nn.Module):
    """
    A transform-average-concatenate (TAC) module.
    """

    def __init__(self, input_size, hidden_size):
        super(TAC, self).__init__()

        self.input_norm = nn.GroupNorm(1, input_size, torch.finfo(torch.float32).eps)
        # self.input_norm = nn.BatchNorm1d(input_size, torch.finfo(torch.float32).eps)
        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size), nn.GELU())
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.TAC_output = nn.Sequential(
            nn.Linear(hidden_size * 2, input_size), nn.GELU()
        )

    def forward(self, input):
        # input shape: batch, group, N, *

        batch_size, G, N = input.shape[:3]
        output = self.input_norm(input.view(batch_size * G, N, -1)).view(
            batch_size, G, N, -1
        )
        T = output.shape[-1]

        # transform
        group_input = output  # B, G, N, T
        group_input = (
            group_input.permute(0, 3, 1, 2).contiguous().view(-1, N)
        )  # B*T*G, N
        group_output = self.TAC_input(group_input).view(
            batch_size, T, G, -1
        )  # B, T, G, H

        # mean pooling
        group_mean = group_output.mean(2).view(batch_size * T, -1)  # B*T, H

        # concate
        group_output = group_output.view(batch_size * T, G, -1)  # B*T, G, H
        group_mean = (
            self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()
        )  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        group_output = self.TAC_output(
            group_output.view(-1, group_output.shape[-1])
        )  # B*T*G, N
        group_output = (
            group_output.view(batch_size, T, G, -1).permute(0, 2, 3, 1).contiguous()
        )  # B, G, N, T
        output = input + group_output.view(input.shape)

        return output


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, bidirectional=True):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps
        if bidirectional:
            self.norm = nn.GroupNorm(1, input_size, self.eps)
        else:
            self.norm = nn.BatchNorm1d(input_size, self.eps)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(
            input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional
        )
        # print(int(bidirectional))
        # linear projection layer
        self.proj = nn.Linear(hidden_size * (int(bidirectional) + 1), input_size)

    def forward(self, input):
        # input shape: batch, dim, seq
        # import pdb; pdb.set_trace()

        rnn_output, _ = self.rnn(
            self.dropout(self.norm(input)).transpose(1, 2).contiguous()
        )
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(input.shape[0], input.shape[2], input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


class gResRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, group=1, dropout=0.0, bidirectional=True
    ):
        super(gResRNN, self).__init__()

        self.group = group
        self.input_size = input_size // group
        self.hidden_size = hidden_size // group
        if input_size % group > 0:
            self.input_size += 1
        if hidden_size % group > 0:
            self.hidden_size += 1

        if group > 1:
            self.comm = TAC(self.input_size, self.input_size * 3)
        else:
            self.comm = nn.Identity()

        self.rnn = ResRNN(self.input_size, self.hidden_size, dropout, bidirectional)

    def forward(self, input):
        # input shape; B, N, T
        # import pdb; pdb.set_trace()
        batch_size, N, T = input.shape

        # split to groups
        if N % self.group > 0:
            # zero padding if necessary
            padding = nn.ConstantPad1d((self.group - N % self.group, 0), 0.0)
            input = padding(input.transpose(1, 2)).transpose(1, 2).contiguous()

        # group comm
        output = self.comm(input.view(batch_size, self.group, -1, T)).view(
            batch_size * self.group, -1, T
        )

        # original module
        output = self.rnn(output).view(batch_size, self.group, -1, T)

        if N % self.group > 0:
            output = output[:, : -(self.group - N % self.group)].contiguous()

        return output


class gMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, group=1):
        super(gMLP, self).__init__()

        self.group = group
        self.input_size = input_size // group
        self.hidden_size = hidden_size // group
        self.output_size = output_size // group
        self.output_rest = output_size % group
        if input_size % group > 0:
            self.input_size += 1
        if hidden_size % group > 0:
            self.hidden_size += 1
        if self.output_rest > 0:
            self.output_size += 1

        if group > 1:
            self.comm = TAC(self.input_size, self.input_size * 3)
        else:
            self.comm = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Conv1d(self.input_size, self.hidden_size, 1),
            nn.Tanh(),
            nn.Conv1d(self.hidden_size, self.hidden_size, 1),
            nn.Tanh(),
            nn.Conv1d(self.hidden_size, self.output_size, 1),
        )

    def forward(self, input):
        # input shape; B, N, T
        batch_size, N, T = input.shape

        # split to groups
        if N % self.group > 0:
            # zero padding if necessary
            padding = nn.ConstantPad1d((self.group - N % self.group, 0), 0.0)
            input = padding(input.transpose(1, 2)).transpose(1, 2).contiguous()

        # group comm
        output = self.comm(input.view(batch_size, self.group, -1, T)).view(
            batch_size * self.group, -1, T
        )

        # original module
        output = self.mlp(output).view(batch_size, self.group, -1, T)

        if self.output_rest > 0:
            output = output[:, : -(self.group - self.output_rest)].contiguous()

        return output


# This is the core block
class BSNet(nn.Module):
    def __init__(
        self,
        in_channel,
        nband=7,
        num_layer=1,
        group=4,
        dropout=0.0,
        bi_comm=True,
        band_bidirectional=True,
    ):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband

        self.band_rnn = []
        for _ in range(num_layer):
            self.band_rnn.append(
                gResRNN(
                    self.feature_dim,
                    self.feature_dim * 2,
                    group,
                    dropout,
                    band_bidirectional,
                )
            )
        self.band_rnn = nn.Sequential(*self.band_rnn)
        self.band_comm = gResRNN(
            self.feature_dim, self.feature_dim * 2, group, dropout, bi_comm
        )
        if band_bidirectional:
            self.output_norm = nn.GroupNorm(
                nband, in_channel, torch.finfo(torch.float32).eps
            )
        else:
            self.output_norm = nn.BatchNorm1d(
                in_channel, torch.finfo(torch.float32).eps
            )

    def forward(self, input):
        # input shape: B, nband*N, T
        # import pdb; pdb.set_trace()
        B, N, T = input.shape

        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim, -1)
        ).view(
            B, self.nband, -1, T
        )  # 各个band 内部的 RNN

        # band comm
        band_output = (
            band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, self.nband)
        )
        output = (
            self.band_comm(band_output)
            .view(B, T, -1, self.nband)
            .permute(0, 3, 2, 1)
            .contiguous()
        )

        return self.output_norm(output.view(B, N, T))


class Separator(nn.Module):
    def __init__(
        self,
        sr=48000,
        win=1536,
        stride=384,
        feature_dim=64,
        num_layer=1,
        num_repeat=3,
        group=8,
        context=1,
        dropout=0.0,
        bi_comm=False,
        band_bidirectional=True,
    ):
        super(Separator, self).__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.context = context
        self.ratio = context * 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        # 0-8k (1k hop), 8k-16k (2k hop), 16k-20k, 20k-inf
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * self.enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.0) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sr / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * self.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (sr / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * self.enc_dim))
        # self.band_width = [bandwidth_100]*10
        # self.band_width += [bandwidth_250]*12

        # v8
        self.band_width = [bandwidth_200] * 20
        self.band_width += [bandwidth_500] * 7

        # bandwidth_1k = int(np.floor(1000 / (sr / 2.) * self.enc_dim))
        # # bandwidth_2k = int(np.floor(2000 / (sr / 2.) * self.enc_dim))
        # # bandwidth_4k = int(np.floor(4000 / (sr / 2.) * self.enc_dim))
        # self.band_width = [bandwidth_1k]*7
        # self.band_width += [bandwidth_2k]*4
        # self.band_width += [bandwidth_4k]
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        print(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            # self.BN.append(nn.Sequential(nn.GroupNorm(1, self.band_width[i]*2, self.eps),
            #                              nn.Conv1d(self.band_width[i]*2, self.feature_dim, 1)
            #                             )
            #               )
            self.BN.append(
                nn.Sequential(
                    nn.BatchNorm1d(self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                )
            )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(
                BSNet(
                    self.nband * self.feature_dim,
                    self.nband,
                    num_layer,
                    group,
                    dropout,
                    bi_comm,
                    band_bidirectional,
                )
            )
        self.separator = nn.Sequential(*self.separator)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                gMLP(
                    self.feature_dim,
                    self.feature_dim * 4,
                    self.band_width[i] * (self.ratio + 1) * 4,
                    group,
                )
            )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size, nch, nsample = input.shape
        input = input.view(batch_size * nch, -1)

        # frequency-domain separation
        spec = torch.stft(
            input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(input.device).type(input.type()),
            return_complex=True,
        )  # B*C, F, T => [1, 257, 126]

        # get a context
        prev_context = []
        post_context = []
        zero_pad = torch.zeros_like(spec)

        for i in range(self.context):
            this_prev_context = torch.cat([zero_pad[:, : i + 1], spec[:, : -1 - i]], 1)
            this_post_context = torch.cat([spec[:, i + 1 :], zero_pad[:, : i + 1]], 1)
            prev_context.append(this_prev_context)
            post_context.append(this_post_context)
        mixture_context = torch.stack(
            prev_context + [spec] + post_context, 1
        )  # B*nch, K, F, T

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_spec_context = []
        subband_power = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(
                spec_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous()
            )
            subband_spec_context.append(
                mixture_context[:, :, band_idx : band_idx + self.band_width[i]]
            )  # B*nch, K, BW, T
            subband_power.append(
                (
                    subband_spec_context[-1].abs().pow(2).mean(1).mean(1) + self.eps
                ).sqrt()
            )  # B*nch, T # context 平均的帧级别能量
            band_idx += self.band_width[i]

        # import pdb; pdb.set_trace()
        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](
                    subband_spec[i].view(batch_size * nch, self.band_width[i] * 2, -1)
                )
            )  # 实部和虚部分开处理
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        # import pdb; pdb.set_trace()

        # separator
        sep_output = self.separator(
            subband_feature.view(batch_size * nch, self.nband * self.feature_dim, -1)
        )  # B, nband*N, T
        sep_output = sep_output.view(batch_size * nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i].contiguous()).view(
                batch_size * nch, 2, 2, self.ratio + 1, self.band_width[i], -1
            )
            this_mask = torch.tanh(this_output[:, 0, :, : self.ratio]) * torch.sigmoid(
                this_output[:, 1, :, : self.ratio]
            )  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = (subband_spec_context[i].real * this_mask_real).mean(1) - (
                subband_spec_context[i].imag * this_mask_imag
            ).mean(
                1
            )  # B*nch, BW, T
            est_spec_imag = (subband_spec_context[i].real * this_mask_imag).mean(1) + (
                subband_spec_context[i].imag * this_mask_real
            ).mean(
                1
            )  # B*nch, BW, T
            this_reg = this_output[:, 0, :, -1] * torch.sigmoid(
                this_output[:, 1, :, -1]
            )  # B*nch, 2, BW, T
            this_reg_real = this_reg[:, 0] * subband_power[i].unsqueeze(
                1
            )  # B*nch, BW, T
            this_reg_imag = this_reg[:, 1] * subband_power[i].unsqueeze(
                1
            )  # B*nch, BW, T
            sep_subband_spec.append(
                torch.complex(
                    est_spec_real + this_reg_real, est_spec_imag + this_reg_imag
                )
            )
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T

        output = torch.istft(
            est_spec.view(batch_size * nch, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(input.device).type(input.type()),
            length=nsample,
        )

        output = output.view(batch_size, nch, -1)

        return output


def check_causal(model):
    input = torch.randn(1, 1, 640 * 32).clamp_(-1, 1)
    fs = 16
    model = model.eval()
    with torch.no_grad():
        out1 = model(input)
        for i in range(640 * 16, 640 * 18, 16):
            inputs2 = input.clone()
            inputs2[..., i:] = 1000 + torch.rand_like(inputs2[..., i:])
            out2 = model(inputs2)
            # import pdb; pdb.set_trace()
            print((i - ((out1 - out2).abs() > 1e-8).float().argmax()) / fs)


if __name__ == "__main__":
    # # import pdb; pdb.set_trace()
    from torchinfo import summary
    from tqdm import tqdm

    model = Separator(
        sr=16000,
        win=512,
        stride=128,
        feature_dim=96,
        num_layer=1,
        num_repeat=6,
        group=1,
        context=1,
        dropout=0.0,
        bi_comm=True,
        band_bidirectional=False,
    )

    x = torch.randn(1, 1, 16000)

    model = model.eval()
    y = model(x)

    summary(model, input_data=x)
