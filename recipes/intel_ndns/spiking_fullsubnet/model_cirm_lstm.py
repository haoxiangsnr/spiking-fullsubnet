from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from audiozen.acoustics.audio_feature import istft, stft


class Model(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        win_length=512,
        input_size=257,
        hidden_size=512,
        num_layers=2,
        proj_size=257,
        **kwargs,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj_size = proj_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.proj = nn.Linear(hidden_size, proj_size * 2)

        self.activation = nn.Tanh()

        self.stft = partial(stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.istft = partial(istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def forward(self, rvb_input):
        # rvb_input: [batch_size, num_samples]
        rvb_mag, rvb_phase, rvb_real, rvb_imag = self.stft(rvb_input)  # [batch_size, num_freqs, num_frames]

        rvb_mag = rearrange(rvb_mag, "b f t -> b t f")
        rnn_output, _ = self.rnn(rvb_mag)  # [batch_size, num_frames, hidden_size]
        rnn_output = self.proj(rnn_output)  # [batch_size, num_frames, proj_size * 2]
        rnn_output = rearrange(rnn_output, "b t f -> b f t")
        rnn_output = self.activation(rnn_output)

        real_out = rnn_output[:, : self.proj_size, :]
        imag_out = rnn_output[:, self.proj_size :, :]

        real_out = real_out * rvb_real - imag_out * rvb_imag
        imag_out = real_out * rvb_imag + imag_out * rvb_real

        y_out = self.istft([real_out, imag_out], length=rvb_input.shape[-1], input_type="real_imag")

        return y_out, rvb_mag, rvb_phase


if __name__ == "__main__":
    import torch

    model = Model()
    x = torch.randn(2, 16000)
    y, mag, phase = model(x)
    print(y.shape, mag.shape, phase.shape)
    print(model)
