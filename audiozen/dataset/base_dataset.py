from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from torch.utils import data
from tqdm import tqdm

from audiozen.acoustics.audio_feature import load_wav


class BaseDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _load_dataset_in_txt(dataset_path, return_empty_if_not_exist=False):
        dataset_path = Path(dataset_path).expanduser().absolute()
        if dataset_path.is_file():
            with open(dataset_path, "r") as f:
                data = [line.rstrip("\n") for line in f]
                return data
        else:
            if return_empty_if_not_exist:
                return []
            else:
                raise FileNotFoundError(f"File {dataset_path} not found.")

    @staticmethod
    def _offset_and_limit(dataset_list, offset, limit):
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]
        return dataset_list

    @staticmethod
    def _parse_snr_range(snr_range):
        assert (
            len(snr_range) == 2
        ), f"The range of SNR should be [low, high], not {snr_range}."
        assert (
            snr_range[0] <= snr_range[-1]
        ), f"The low SNR should not larger than high SNR."

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)

        return snr_list

    def _preload_dataset(self, file_path_list, remark=""):
        waveform_list = Parallel(n_jobs=self.num_workers)(
            delayed(self._load_wav)(f_path, sr=self.sr)
            for f_path in tqdm(file_path_list, desc=remark)
        )
        return list(zip(file_path_list, waveform_list))  # type: ignore

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_path_list)
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
