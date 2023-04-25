import os
from pathlib import Path

import librosa

from audiozen.acoustics.audio_feature import load_wav
from audiozen.dataset.base_dataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, dataset_dir_list, sr, offset=0, limit=None):
        super(Dataset, self).__init__()
        noisy_files_list = []

        for dataset_dir in dataset_dir_list:
            dataset_dir = Path(dataset_dir).expanduser().absolute()
            noisy_files_list += librosa.util.find_files(dataset_dir.as_posix())

        if offset > 0:
            noisy_files_list = noisy_files_list[offset:]

        if limit:
            noisy_files_list = noisy_files_list[:limit]

        self.length = len(noisy_files_list)
        self.noisy_files_list = noisy_files_list
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_file_path = self.noisy_files_list[item]
        noisy = load_wav(os.path.abspath(os.path.expanduser(noisy_file_path)), sr=self.sr)
        return noisy, noisy_file_path
