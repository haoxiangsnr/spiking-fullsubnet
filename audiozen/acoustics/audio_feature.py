import os
from typing import Literal, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from torch import Tensor

from audiozen.constant import EPSILON


def is_audio(y_s: list[NDArray] | NDArray):
    if not isinstance(y_s, list):
        y_s = [y_s]

    for y in y_s:
        assert y.ndim in (1, 2), "Only support signals with the shape of [C, T] or [T]."


def compute_rms(y: NDArray) -> float:
    """Compute the Root Mean Square (RMS) of the given signal."""
    return np.sqrt(np.mean(y**2))


def loudness_max_norm(
    y: NDArray,
    scalar: float | None = None,
    ref_mic: int = 0,
    eps: float = EPSILON,
) -> tuple[NDArray, float]:
    """Maximum loudness normalization to signals."""
    if not scalar:
        if y.ndim == 1:
            scalar = np.max(np.abs(y)) + eps
        else:
            scalar = np.max(np.abs(y[ref_mic, :])) + eps

    assert scalar is not None
    return y / scalar, scalar


def loudness_rms_norm(
    y: NDArray,
    scalar: float | None = None,
    lvl: float = -25,
    ref_mic: int = 0,
    eps: float = EPSILON,
) -> tuple[NDArray, float]:
    """Loudness normalize a signal based on the Root Mean Square (RMS).

    Normalize the RMS of signals to a given RMS based on Decibels Relative to Full Scale (dBFS).

    Args:
        y: [C, T] or [T,].
        scalar: scalar to normalize the RMS, default to None.
        target_rms: target RMS in dBFS.
        ref_mic: reference mic for multi-channel signals.

    Returns:
        Loudness normalized signal and scalar.

    Note:
        A small amount of signal samples would be clipped after normalization, but it does not matter.
    """
    if not scalar:
        current_level = compute_rms(y) if y.ndim == 1 else compute_rms(y[ref_mic, :])
        scalar = 10 ** (lvl / 20) / (current_level + eps)

    return y * scalar, scalar


def sxr2gain(
    meaningful: NDArray,
    meaningless: NDArray,
    desired_ratio: float,
    eps: float = EPSILON,
) -> float:
    """Generally calculate the gains of interference to fulfill a desired SXR (SNR or SIR) ratio.

    Args:
        meaningful: meaningful input, like target clean.
        meaningless: meaningless or unwanted input, like background noise.
        desired_ratio: SNR or SIR ratio.

    Returns:
        Gain, which can be used to adjust the RMS of the meaningless signals to satisfy the given ratio.
    """
    meaningful_rms = compute_rms(meaningful)
    meaningless_rms = compute_rms(meaningless)
    scalar = meaningful_rms / (10 ** (desired_ratio / 20)) / (meaningless_rms + eps)

    return scalar


def load_wav(file: str, sr: int = 16000) -> NDArray:
    """Load a wav file.

    Args:
        file: file path.
        sr: sample rate. Defaults to 16000.

    Returns:
        Waveform with shape of [C, T] or [T].
    """
    return librosa.load(os.path.abspath(os.path.expanduser(file)), mono=False, sr=sr)[
        0
    ]  # [C, T] or [T]


def save_wav(data, fpath, sr):
    if data.ndim != 1:
        data = data.reshape(-1)

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    sf.write(fpath, data, sr)


def mag_phase(complex_valued_tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Get magnitude and phase of a complex-valued tensor.

    Args:
        complex_valued_tensor: complex-valued tensor.

    Returns:
        magnitude and phase spectrogram.
    """
    mag, phase = torch.abs(complex_valued_tensor), torch.angle(complex_valued_tensor)
    return mag, phase


def stft(
    y: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    output_type: Literal["mag_phase", "real_imag", "complex"] | None = None,
) -> Tensor | tuple[Tensor, ...]:
    """Wrapper of the official ``torch.stft`` for single-channel and multichannel signals.

    Args:
        y: single-/multichannel signals with shape of [B, C, T] or [B, T].
        n_fft: num of FFT.
        hop_length: hop length.
        win_length: hanning window size.
        output_type: "mag_phase", "real_imag", "complex", or None.

    Returns:
        If the input is single-channel, return the spectrogram with shape of [B, F, T], otherwise [B, C, F, T].
        If output_type is "mag_phase", return a list of magnitude and phase spectrogram.
        If output_type is "real_imag", return a list of real and imag spectrogram.
        If output_type is None, return a list of magnitude, phase, real, and imag spectrogram.
    """
    ndim = y.dim()
    assert ndim in [2, 3], f"Only support single/multi-channel signals. {ndim=}."

    batch_size, *_, num_samples = y.shape

    if ndim == 3:
        # single-channel
        y = y.reshape(-1, num_samples)

    complex_valued_stft = torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=y.device),
        return_complex=True,
    )

    _, num_freqs, num_frames = complex_valued_stft.shape

    if ndim == 3:
        # single-channel
        complex_valued_stft = complex_valued_stft.reshape(
            batch_size, -1, num_freqs, num_frames
        )

    if output_type:
        match output_type:
            case "mag_phase":
                return mag_phase(complex_valued_stft)
            case "real_imag":
                return complex_valued_stft.real, complex_valued_stft.imag
            case "complex":
                return complex_valued_stft
            case _:
                raise NotImplementedError(
                    "Only 'mag_phase', 'real_imag', and 'complex' are supported"
                )
    else:
        return (
            *mag_phase(complex_valued_stft),
            complex_valued_stft.real,
            complex_valued_stft.imag,
        )


def istft(
    feature: Tensor | tuple[Tensor, ...] | list[Tensor],
    n_fft: int,
    hop_length: int,
    win_length: int,
    length: Optional[int] = None,
    input_type: Literal["mag_phase", "real_imag", "complex"] = "mag_phase",
) -> Tensor:
    """Wrapper of the official ``torch.istft`` for single-channel signals.

    Args:
        features: single-channel spectrogram with shape of [B, F, T] for input_type="complex" or ([B, F, T], [B, F, T]) for input_type="real_imag" and "mag_phase".
        n_fft: num of FFT.
        hop_length: hop length.
        win_length: hanning window size.
        length: expected length of istft.
        input_type: "real_image", "complex", or "mag_phase".

    Returns:
        Single-channel singal with the shape shape of [B, T].

    Notes:
        Only support single-channel input with shape of [B, F, T] or ([B, F, T], [B, F, T])
    """
    if input_type == "real_imag":
        assert isinstance(feature, tuple) or isinstance(
            feature, list
        )  # (real, imag) or [real, imag]
        real, imag = feature
        complex_valued_features = torch.complex(real=real, imag=imag)
    elif input_type == "complex":
        assert isinstance(feature, Tensor) and torch.is_complex(feature)
        complex_valued_features = feature
    elif input_type == "mag_phase":
        assert isinstance(feature, tuple) or isinstance(
            feature, list
        )  # (mag, phase) or [mag, phase]
        mag, phase = feature
        complex_valued_features = torch.complex(
            mag * torch.cos(phase), mag * torch.sin(phase)
        )
    else:
        raise NotImplementedError(
            "Only 'real_imag', 'complex', and 'mag_phase' are supported now."
        )

    return torch.istft(
        complex_valued_features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=complex_valued_features.device),
        length=length,
    )


def norm_amplitude(y: NDArray, scalar: Optional[float] = None, eps: float = EPSILON):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def subsample(
    data: NDArray,
    sub_sample_length: int,
    start_position: int = -1,
    return_start_position: bool = False,
) -> NDArray | tuple[NDArray, int]:
    """Sample a segment from the input data.

    Args:
        data: **one-dimensional data**
        sub_sample_length: The length of the segment to be sampled
        start_position: The start index of the segment to be sampled. If start_position smaller than 0, randomly generate one index

    """
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        if start_position < 0:
            start_position = np.random.randint(length - sub_sample_length)
        end = start_position + sub_sample_length
        data = data[start_position:end]
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
    else:
        pass

    assert len(data) == sub_sample_length

    if return_start_position:
        return data, start_position
    else:
        return data


def is_clipped(y: NDArray, clipping_threshold: float = 0.999) -> bool:
    """Check if the input signal is clipped."""
    return (np.abs(y) > clipping_threshold).any()  # type: ignore


def tune_dB_FS(y, target_dB_FS=-26, eps=EPSILON):
    """Tune the RMS of the input signal to a target level.

    Args:
        y: Audio signal with any shape.
        target_dB_FS: Target dB_FS. Defaults to -25.
        eps: A small value to avoid dividing by zero. Defaults to EPSILON.

    Returns:
        Scaled audio signal with the same shape as the input.
    """
    if isinstance(y, torch.Tensor):
        rms = torch.sqrt(torch.mean(y**2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar
    else:
        rms = np.sqrt(np.mean(y**2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar


def activity_detector(
    audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=EPSILON
):
    """Return the percentage of the time the audio signal is above an energy threshold.

    Args:
        audio:
        fs:
        activity_threshold:
        target_level:
        eps:

    Returns:

    """
    audio, _, _ = tune_dB_FS(audio, target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win**2) + eps)
        frame_energy_prob = 1.0 / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (
                1 - alpha_att
            )
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (
                1 - alpha_rel
            )

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def build_complex_ideal_ratio_mask(
    noisy_real, noisy_imag, clean_real, clean_imag
) -> torch.Tensor:
    """Build the complex ratio mask.

    Args:
        noisy: [B, F, T], noisy complex-valued stft coefficients
        clean: [B, F, T], clean complex-valued stft coefficients

    References:
        https://ieeexplore.ieee.org/document/7364200

    Returns:
        [B, F, T, 2]
    """
    denominator = torch.square(noisy_real) + torch.square(noisy_imag) + EPSILON

    mask_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denominator
    mask_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denominator

    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)

    return compress_cIRM(complex_ratio_mask, K=10, C=0.1)


def compress_cIRM(mask, K=10, C=0.1):
    """Compress the value of cIRM from (-inf, +inf) to [-K ~ K].

    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def decompress_cIRM(mask, K=10, limit=9.9):
    """Decompress cIRM from [-K ~ K] to [-inf, +inf].

    Args:
        mask: cIRM mask
        K: default 10
        limit: default 0.1

    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    mask = (
        limit * (mask >= limit)
        - limit * (mask <= -limit)
        + mask * (torch.abs(mask) < limit)
    )
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


def complex_mul(noisy_r, noisy_i, mask_r, mask_i):
    r = noisy_r * mask_r - noisy_i * mask_i
    i = noisy_r * mask_i + noisy_i * mask_r
    return r, i


class Mask:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_mask(mask, title="Mask", xlim=None):
        mask = mask.numpy()
        figure, axis = plt.subplots(1, 1)
        img = axis.imshow(mask, cmap="viridis", origin="lower", aspect="auto")
        figure.suptitle(title)
        plt.colorbar(img, ax=axis)
        plt.show()


class IRM(Mask):
    def __init__(self) -> None:
        super(IRM, self).__init__()

    @staticmethod
    def generate_mask(clean, noise, ref_channel=0):
        """Generate an ideal ratio mask.

        Args:
            clean: Complex STFT of clean audio with the shape of [B, C, F, T].
            noise: Complex STFT of noise audio with the same shape of [B, 1, F, T].
            ref_channel: The reference channel to compute the mask if the STFTs are multi-channel.

        Returns:
            Speech mask and noise mask with the shape of [B, 1, F, T].
        """
        mag_clean = clean.abs() ** 2
        mag_noise = noise.abs() ** 2
        irm_speech = mag_clean / (mag_clean + mag_noise)
        irm_noise = mag_noise / (mag_clean + mag_noise)

        if irm_speech.ndim == 4:
            irm_speech = irm_speech[ref_channel, :][:, None, ...]
            irm_noise = irm_noise[ref_channel, :][:, None, ...]

        return irm_speech, irm_noise


def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert (
        batch_size > num_groups
    ), f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., : (num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(
            group_idx, batch_size, num_groups, device=input.device
        )
        freqs_indices = torch.arange(
            group_idx, num_freqs, num_groups, device=input.device
        )

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(
            selected_samples, dim=2, index=freqs_indices
        )  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)
