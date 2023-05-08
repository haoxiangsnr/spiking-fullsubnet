from typing import Union

import numpy as np
import torch
from lava.lib.dl import slayer  # type: ignore

from audiozen.trainer.base_trainer import BaseTrainer


def si_snr(
    target: Union[torch.tensor, np.ndarray],
    estimate: Union[torch.tensor, np.ndarray],
    EPS=1e-8,
) -> torch.tensor:
    """Calculates SI-SNR estiamte from target audio and estiamte audio. The
    audio sequene is expected to be a tensor/array of dimension more than 1.
    The last dimension is interpreted as time.

    The implementation is based on the example here:
    https://www.tutorialexample.com/wp-content/uploads/2021/12/SI-SNR-definition.png

    Parameters
    ----------
    target : Union[torch.tensor, np.ndarray]
        Target audio waveform.
    estimate : Union[torch.tensor, np.ndarray]
        Estimate audio waveform.

    Returns
    -------
    torch.tensor
        SI-SNR of each target and estimate pair.
    """
    if not torch.is_tensor(target):
        target: torch.tensor = torch.tensor(target)
    if not torch.is_tensor(estimate):
        estimate: torch.tensor = torch.tensor(estimate)

    # zero mean to ensure scale invariance
    s_target = target - torch.mean(target, dim=-1, keepdim=True)
    s_estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target**2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=-1) / (
        torch.sum(e_noise**2, dim=-1) + EPS
    )
    return 10 * torch.log10(pair_wise_sdr + EPS)


def stft_splitter(audio, n_fft=512):
    with torch.no_grad():
        audio_stft = torch.stft(audio, n_fft=n_fft, onesided=True, return_complex=True)
        return audio_stft.abs(), audio_stft.angle()


def stft_mixer(stft_abs, stft_angle, n_fft=512):
    return torch.istft(
        torch.complex(
            stft_abs * torch.cos(stft_angle), stft_abs * torch.sin(stft_angle)
        ),
        n_fft=n_fft,
        onesided=True,
    )


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        noisy, clean, noise = batch

        out_delay = 0

        noisy = noisy.to(self.device)
        clean = clean.to(self.device)

        noisy_abs, noisy_arg = stft_splitter(noisy, self.n_fft)  # [B, F, T]
        clean_abs, clean_arg = stft_splitter(clean, self.n_fft)  # [B, F, T]

        denoised_abs = self.model(noisy_abs)
        noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
        clean_abs = slayer.axon.delay(clean_abs, out_delay)
        clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

        clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft)

        score = si_snr(clean_rec, clean)
        loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

        return loss
