import random

import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from tqdm import tqdm

from audiozen.acoustics.audio_feature import (
    is_clipped,
    load_wav,
    norm_amplitude,
    subsample,
    tune_dB_FS,
)
from audiozen.dataset.base_dataset import BaseDataset
from audiozen.utils import expand_path


class Dataset(BaseDataset):
    def __init__(
        self,
        clean_dataset,
        clean_dataset_limit,
        clean_dataset_offset,
        noise_dataset,
        noise_dataset_limit,
        noise_dataset_offset,
        rir_dataset,
        rir_dataset_limit,
        rir_dataset_offset,
        snr_range,
        reverb_proportion,
        silence_length,
        target_dB_FS,
        target_dB_FS_floating_value,
        sub_sample_length,
        sr,
        pre_load_clean_dataset,
        pre_load_noise,
        pre_load_rir,
        num_workers,
        learning_target="full",
        len_early_rir=None,
    ):
        """Dynamic generate mixing data for training"""
        super().__init__()
        # acoustics args
        self.sr = sr

        # parallel args
        self.num_workers = num_workers

        clean_dataset_list = [
            line.rstrip("\n") for line in open(expand_path(clean_dataset), "r")
        ]
        noise_dataset_list = [
            line.rstrip("\n") for line in open(expand_path(noise_dataset), "r")
        ]
        rir_dataset_list = [
            line.rstrip("\n") for line in open(expand_path(rir_dataset), "r")
        ]

        clean_dataset_list = self._offset_and_limit(
            clean_dataset_list, clean_dataset_offset, clean_dataset_limit
        )
        noise_dataset_list = self._offset_and_limit(
            noise_dataset_list, noise_dataset_offset, noise_dataset_limit
        )
        rir_dataset_list = self._offset_and_limit(
            rir_dataset_list, rir_dataset_offset, rir_dataset_limit
        )

        if pre_load_clean_dataset:
            clean_dataset_list = self._preload_dataset(
                clean_dataset_list, remark="Clean Dataset"
            )

        if pre_load_noise:
            noise_dataset_list = self._preload_dataset(
                noise_dataset_list, remark="Noise Dataset"
            )

        if pre_load_rir:
            rir_dataset_list = self._preload_dataset(
                rir_dataset_list, remark="RIR Dataset"
            )

        self.clean_dataset_list = clean_dataset_list
        self.noise_dataset_list = noise_dataset_list
        self.rir_dataset_list = rir_dataset_list

        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list

        assert (
            0 <= reverb_proportion <= 1
        ), "The 'reverb_proportion' should be in [0, 1]."
        self.reverb_proportion = reverb_proportion
        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.sub_sample_length = sub_sample_length
        self.learning_target = learning_target
        self.len_early_rir = len_early_rir

        self.length = len(self.clean_dataset_list)

    def __len__(self):
        return self.length

    def _preload_dataset(self, file_path_list, remark=""):
        waveform_list = Parallel(n_jobs=self.num_workers)(
            delayed(load_wav)(f_path, sr=self.sr)
            for f_path in tqdm(file_path_list, desc=remark)
        )
        return list(zip(file_path_list, waveform_list))  # type: ignore

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)
            noise_new_added = load_wav(noise_file, sr=self.sr)
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            # If still need to add new noise, insert a small silence segment firstly
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start : idx_start + target_length]

        return noise_y

    @staticmethod
    def find_peak_idx(rir):
        """Find the peak index of the RIR.

        Args:
            rir: room impulse response with the shape of [T] or [C, T], where C is the number of channels.

        Returns:
            peak index
        """
        return np.argmax(np.abs(rir))

    @staticmethod
    def random_select_channel(rir):
        """Randomly select a channel of the RIR.

        Args:
            rir: room impulse response with the shape of [C, T], where C is the number of channels.

        Returns:
            selected channel of the RIR
        """
        return rir[np.random.randint(0, rir.shape[0])]

    def conv_rir(self, clean_y, rir, learning_target="early"):
        """Convolve clean_y with a RIR.

        Args:
            clean_y: clean signal.
            rir: room impulse response with the shape of [T] or [C, T], where C is the number of channels.
            learning_target: the learning target, which can be 'direct_path', 'early', or 'full'.

        Returns:
            convolved signal
        """
        assert rir.ndim in [1, 2], "The dimension of the RIR should be 1 or 2."
        assert learning_target in [
            "direct_path",
            "early",
            "full",
        ], "The learning target should be 'direct_path' or 'early', or 'full'."

        if rir.ndim == 2:
            rir = self.random_select_channel(rir)

        # Find the RIR of the learning target
        dp_idx = self.find_peak_idx(rir)

        if learning_target == "direct_path":
            rir = rir[:dp_idx]
        elif learning_target == "early":
            assert self.len_early_rir is not None, "The 'len_early_rir' should be set."
            len_early_rir = int(self.sr * self.len_early_rir)
            rir = rir[: dp_idx + len_early_rir]
        elif learning_target == "full":
            rir = rir
        else:
            raise ValueError(
                "The learning target should be 'direct_path' or 'early', or 'full'."
            )

        # audio with full-length RIR
        clean_y_rvb = signal.fftconvolve(clean_y, rir)
        clean_y_rvb = clean_y_rvb[: len(clean_y)]

        return clean_y_rvb

    def snr_mix(
        self,
        clean_y,
        noise_y,
        snr,
        target_dB_FS,
        target_dB_FS_floating_value,
        rir=None,
        eps=1e-6,
    ):
        """Mix clean_y and noise_y based on a given SNR and a RIR (if exist).

        Args:
            clean_y: clean signal
            noise_y: noise signal
            snr (int): signal-to-noise ratio
            target_dB_FS (int): target dBFS
            target_dB_FS_floating_value (int): target dBFS floating value
            rir: room impulse response. It can be None or a numpy.ndarray
            eps: eps

        Returns:
            (noisy_y, clean_y)
        """
        if rir is not None:
            clean_y_t60_rvb = self.conv_rir(clean_y, rir, "full")
            clean_y_target_rvb = self.conv_rir(clean_y, rir, self.learning_target)
        else:
            clean_y_t60_rvb = clean_y
            clean_y_target_rvb = clean_y

        clean_y_t60_rvb, _ = norm_amplitude(clean_y_t60_rvb)
        clean_y_t60_rvb, _, _ = tune_dB_FS(clean_y_t60_rvb, target_dB_FS)
        clean_y_t60_rvb_rms = (clean_y_t60_rvb**2).mean() ** 0.5

        clean_y_target_rvb, _ = norm_amplitude(clean_y_target_rvb)
        clean_y_target_rvb, _, _ = tune_dB_FS(clean_y_target_rvb, target_dB_FS)

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tune_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y**2).mean() ** 0.5

        snr_scalar = clean_y_t60_rvb_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y_t60_rvb + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value,
        )
        noisy_y, _, noisy_scalar = tune_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y_target_rvb *= noisy_scalar

        # The mixed speech is clipped if the RMS value of noisy speech is too large.
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)
            noisy_y = noisy_y / noisy_y_scalar
            clean_y_target_rvb = clean_y_target_rvb / noisy_y_scalar

        return noisy_y, clean_y_target_rvb

    def __getitem__(self, item):
        clean_fpath = self.clean_dataset_list[item]
        clean_y = load_wav(clean_fpath, sr=self.sr)
        clean_y = subsample(
            data=clean_y,
            sub_sample_length=int(self.sub_sample_length * self.sr),
        )
        noise_y = self._select_noise_y(target_length=len(clean_y))
        if len(noise_y) != len(clean_y):
            raise ValueError(f"Inequality: {len(clean_y)=} {len(noise_y)=}")

        snr = self._random_select_from(self.snr_list)
        use_reverb = bool(np.random.random(1) < self.reverb_proportion)

        noisy_y, clean_y = self.snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value,
            rir=load_wav(self._random_select_from(self.rir_dataset_list), sr=self.sr)
            if use_reverb
            else None,
        )

        noisy_y = noisy_y.astype(np.float32)
        clean_y = clean_y.astype(np.float32)

        return noisy_y, clean_y
