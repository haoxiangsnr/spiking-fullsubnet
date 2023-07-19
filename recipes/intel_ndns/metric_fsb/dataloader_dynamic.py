import os

import numpy as np

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
        sr,
        silence_length=0.2,
        target_dB_FS=-26.0,
        target_dB_FS_floating_value=10.0,
        sub_sample_length=6,
    ):
        super().__init__()
        clean_fpath_list = [line.rstrip("\n") for line in open(clean_dataset, "r")]

        noise_fpath_list = [line.rstrip("\n") for line in open(noise_dataset, "r")]

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
        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.sub_sample_length = sub_sample_length

    def segmental_snr_mixer(self, clean, noise, snr):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, item):
        pass
