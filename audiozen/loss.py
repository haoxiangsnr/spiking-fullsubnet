"""Loss module includes loss functions related to speech signal processing."""
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torch.autograd import Function

from audiozen.constant import EPSILON


class SISNRLoss:
    def __init__(self, EPS=1e-8) -> None:
        self.EPS = EPS

    def forward(self, input: Tensor | ndarray, target: Tensor | ndarray):
        if isinstance(input, ndarray):
            input = torch.from_numpy(input)
        if isinstance(target, ndarray):
            target = torch.from_numpy(target)

        if input.shape != target.shape:
            raise RuntimeError(
                f"Dimension mismatch when calculate si_snr, {input.shape} vs {target.shape}"
            )

        s_input = input - torch.mean(input, dim=-1, keepdim=True)
        s_target = target - torch.mean(target, dim=-1, keepdim=True)

        # <s, s'> / ||s||**2 * s
        pair_wise_dot = torch.sum(s_target * s_input, dim=-1, keepdim=True)
        s_target_norm = torch.sum(s_target**2, dim=-1, keepdim=True)
        pair_wise_proj = pair_wise_dot * s_target / s_target_norm

        e_noise = s_input - pair_wise_proj

        pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=-1) / (
            torch.sum(e_noise**2, dim=-1) + self.EPS
        )
        return 10 * torch.log10(pair_wise_sdr + self.EPS)


def si_snr_loss():
    def si_snr(x, s, eps=EPSILON):
        """

        Args:
            x: Enhanced fo shape [B, T]
            s: Reference of shape [B, T]
            eps:

        Returns:
            si_snr: [B]
        """

        def l2norm(mat, keep_dim=False):
            return torch.norm(mat, dim=-1, keepdim=keep_dim)

        if x.shape != s.shape:
            raise RuntimeError(
                f"Dimension mismatch when calculate si_snr, {x.shape} vs {s.shape}"
            )

        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)

        t = (
            torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
            * s_zm
            / (l2norm(s_zm, keep_dim=True) ** 2 + eps)
        )

        return -torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))

    return si_snr


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(EPSILON)
        return torch.view_as_complex(
            torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1)
        )


class MultiResSpecLoss(nn.Module):
    def __init__(self, n_ffts, gamma=1, factor=1, f_complex=None):
        super().__init__()
        self.n_fft_list = n_ffts
        self.gamma = gamma
        self.factor = factor
        self.factor_complex = f_complex

    @staticmethod
    def stft(y, n_fft, hop_length=None, win_length=None):
        num_dims = y.dim()
        assert num_dims == 2, "Only support 2D input."

        hop_length = hop_length or n_fft // 4
        win_length = win_length or n_fft

        batch_size, num_samples = y.size()

        # [B, F, T] in complex-valued
        complex_stft = torch.stft(
            y,
            n_fft,
            hop_length,
            win_length,
            window=torch.hann_window(n_fft, device=y.device),
            return_complex=True,
            normalized=True,
        )

        return complex_stft

    def forward(self, est: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=est.device, dtype=est.dtype)

        for n_fft in self.n_fft_list:
            Y = self.stft(est, n_fft)
            S = self.stft(target, n_fft)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(EPSILON).pow(self.gamma)
                S_abs = S_abs.clamp_min(EPSILON).pow(self.gamma)

            # magnitude loss
            loss += F.mse_loss(Y_abs, S_abs) * self.factor

            # real/imaginary loss
            if self.factor_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += (
                    F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S))
                    * self.factor_complex
                )
        return loss


class CombineLoss(nn.Module):
    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: float = None,
    ):
        super().__init__()
        self.n_ffts = n_ffts
        self.gamma = gamma
        self.f = factor
        self.f_complex = f_complex

        self.mulres_loss = MultiResSpecLoss(n_ffts, gamma, factor, f_complex)
        self.l1_loss = nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss1 = self.mulres_loss(input, target)
        loss2 = self.l1_loss(input, target)
        return loss1 + loss2


def freq_MAE(estimation, target, win=2048, stride=512, srs=None, sudo_sr=None):
    est_spec = torch.stft(
        estimation.view(-1, estimation.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )
    est_target = torch.stft(
        target.view(-1, target.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )

    if srs is None:
        return (est_spec.real - est_target.real).abs().mean() + (
            est_spec.imag - est_target.imag
        ).abs().mean()
    else:
        loss = 0
        for i, sr in enumerate(srs):
            max_freq = int(est_spec.shape[-2] * sr / sudo_sr)
            loss += (
                est_spec[i][:max_freq].real - est_target[i][:max_freq].real
            ).abs().mean() + (
                est_spec[i][:max_freq].imag - est_target[i][:max_freq].imag
            ).abs().mean()
        loss = loss / len(srs)
        return loss


def mag_MAE(estimation, target, win=2048, stride=512, srs=None, sudo_sr=None):
    est_spec = torch.stft(
        estimation.view(-1, estimation.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )
    est_target = torch.stft(
        target.view(-1, target.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )
    if srs is None:
        return (est_spec.abs() - est_target.abs()).abs().mean()
    else:
        loss = 0
        for i, sr in enumerate(srs):
            max_freq = int(est_spec.shape[-2] * sr / sudo_sr)
            loss += (
                (est_spec[i][:max_freq].abs() - est_target[i][:max_freq].abs())
                .abs()
                .mean()
            )
        loss = loss / len(srs)
    return loss


if __name__ == "__main__":
    import torch

    n_ffts = [240, 480, 960, 1440]
    gamma = 0.3
    factor = 1
    f_complex = 1

    loss = CombineLoss(n_ffts, gamma, factor, f_complex)
    input = torch.rand(2, 16000)
    target = torch.rand(2, 16000)
    print(loss(input, target))
