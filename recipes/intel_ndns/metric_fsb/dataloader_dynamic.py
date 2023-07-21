import random

import numpy as np

from audiozen.acoustics.audio_feature import (
    active_rms,
    is_clipped,
    loudness_max_norm,
    loudness_rms_norm,
    normalize_segmental_rms,
)
from audiozen.constant import EPSILON
from audiozen.dataset.base_dataset import BaseDataset


class DNSAudio(BaseDataset):
    def __init__(
        self,
        clean_dataset,
        clean_dataset_limit,
        clean_dataset_offset,
        noise_dataset,
        noise_dataset_limit,
        noise_dataset_offset,
        snr_range,
        sr=16000,
        dataset_length=60000,
        loudness_lvl=-25,
        loudness_floating_value=10,
        sample_length=6,
    ):
        super().__init__()
        clean_fpath_list = self._load_dataset_from_text_and_dir_list(clean_dataset)
        print(f"Founds {len(clean_fpath_list)} clean files")
        noise_fpath_list = self._load_dataset_from_text_and_dir_list(noise_dataset)
        print(f"Founds {len(noise_fpath_list)} noise files")

        clean_fpath_list = self._offset_and_limit(
            clean_fpath_list, clean_dataset_offset, clean_dataset_limit
        )

        noise_fpath_list = self._offset_and_limit(
            noise_fpath_list, noise_dataset_offset, noise_dataset_limit
        )

        self.sr = sr
        self.clean_fpath_list = clean_fpath_list
        self.noise_fpath_list = noise_fpath_list
        self.snr_list = self._parse_snr_range(snr_range)
        self.loudness_lvl = loudness_lvl
        self.loudness_floating_value = loudness_floating_value
        self.sample_length = sample_length
        self.dataset_length = dataset_length

        print(f"Each epoch will have {self.dataset_length} samples.")

    def __len__(self):
        return self.dataset_length

    def simulate_mixture(
        self,
        clean_fpath,
        sample_length,
        snr,
        loudness_lvl,
        loudness_floating_value,
        sr=16000,
        clipping_threshold=0.99,
    ):
        clean_y, _ = self._load_wav(
            clean_fpath, duration=sample_length, sr=sr, mode="constant"
        )
        noise_path = random.choice(self.noise_fpath_list)
        noise_y, _ = self._load_wav(
            noise_path, duration=sample_length, sr=sr, mode="wrap"
        )

        clean_y, _ = loudness_max_norm(clean_y)
        noise_y, _ = loudness_max_norm(noise_y)

        clean_rms, noise_rms = active_rms(clean=clean_y, noise=noise_y)

        clean_y = normalize_segmental_rms(clean_y, clean_rms, loudness_lvl)
        noise_y = normalize_segmental_rms(noise_y, noise_rms, loudness_lvl)

        scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + EPSILON)
        scaled_noise_y = noise_y * scalar

        mix_y = clean_y + scaled_noise_y

        mix_lvl = random.randint(
            loudness_lvl - loudness_floating_value,
            loudness_lvl + loudness_floating_value,
        )
        mix_y, mix_scalar = loudness_rms_norm(mix_y, lvl=mix_lvl)
        clean_y = clean_y * mix_scalar
        scaled_noise_y = scaled_noise_y * mix_scalar

        if is_clipped(mix_y):
            mix_maxamplevel = max(abs(mix_y)) / (clipping_threshold - EPSILON)
            mix_y = mix_y / mix_maxamplevel
            clean_y = clean_y / mix_maxamplevel
            scaled_noise_y = scaled_noise_y / mix_maxamplevel

        return mix_y, clean_y, scaled_noise_y

    def __getitem__(self, _):
        clean_fpath = random.choice(self.clean_fpath_list)
        snr = random.choice(self.snr_list)

        mix_y, clean_y, noise_y = self.simulate_mixture(
            clean_fpath,
            sample_length=self.sample_length,
            snr=snr,
            loudness_lvl=self.loudness_lvl,
            loudness_floating_value=self.loudness_floating_value,
            sr=self.sr,
        )

        # Return the mixed audio and the SNR
        mix_y = mix_y.astype(np.float32)
        clean_y = clean_y.astype(np.float32)
        noise_y = noise_y.astype(np.float32)

        return mix_y, clean_y, noise_y


if __name__ == "__main__":
    import soundfile as sf

    dataset = DNSAudio(
        clean_dataset="/datasets/datasets_fullband/datasets_fullband/clean_fullband",
        clean_dataset_limit=None,
        clean_dataset_offset=0,
        noise_dataset="/datasets/datasets_fullband/datasets_fullband/noise_fullband",
        noise_dataset_limit=None,
        noise_dataset_offset=0,
        snr_range=[-5, 20],
        sr=16000,
        loudness_lvl=-25,
        loudness_floating_value=10,
        sample_length=6,
    )

    for i in range(10):
        mix_y, clean_y, noise_y = dataset[i]
        sf.write(f"tmp/mix_{i}.wav", mix_y, samplerate=16000)
        sf.write(f"tmp/clean_{i}.wav", clean_y, samplerate=16000)
        sf.write(f"tmp/noise_{i}.wav", noise_y, samplerate=16000)
