import torch.nn as nn
import torch.nn.functional as F

from audiozen.acoustics.audio_feature import stft
from audiozen.loss import SISNRLoss


class SDNNLoss(nn.Module):
    def __init__(self, lam, n_fft, hop_length, win_length) -> None:
        """Intel N-DNS baselie loss

        Args:
            lam: lagrangian factor
            n_fft: fft window size
            hop_length: hop length
            win_length: window length
        """
        super().__init__()
        self.si_snr = SISNRLoss()
        self.lam = lam
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, input, target):
        score = self.si_snr(input, target)

        # input_mag, *_ = stft(input, self.n_fft, self.hop_length, self.win_length)
        # target_mag, *_ = stft(target, self.n_fft, self.hop_length, self.win_length)

        # loss = self.lam * F.mse_loss(input_mag, target_mag) + (100 - score)

        return 100 - score
