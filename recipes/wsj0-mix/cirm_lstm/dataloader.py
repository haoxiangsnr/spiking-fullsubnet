from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from torch.utils import data

from audiozen.acoustics.io import subsample


class Dataset(data.Dataset):
    def __init__(
        self, mix_scp_or_dir, s1_scp_or_dir, s2_scp_or_dir, sr=8000, duration=4, is_train=True, limit=None, offset=0
    ):
        super().__init__()
        mix_scp_or_dir = Path(mix_scp_or_dir).expanduser().resolve()
        s1_scp_or_dir = Path(s1_scp_or_dir).expanduser().resolve()
        s2_scp_or_dir = Path(s2_scp_or_dir).expanduser().resolve()

        if mix_scp_or_dir.is_dir():
            mix_fpath_list = librosa.util.find_files(mix_scp_or_dir)
            s1_fpath_list = librosa.util.find_files(s1_scp_or_dir)
            s2_fpath_list = librosa.util.find_files(s2_scp_or_dir)
        else:
            with open(mix_scp_or_dir, "r") as f:
                mix_fpath_list = f.read().splitlines()
            with open(s1_scp_or_dir, "r") as f:
                s1_fpath_list = f.read().splitlines()
            with open(s2_scp_or_dir, "r") as f:
                s2_fpath_list = f.read().splitlines()

        if offset > 0:
            mix_fpath_list = mix_fpath_list[offset:]
            s1_fpath_list = s1_fpath_list[offset:]
            s2_fpath_list = s2_fpath_list[offset:]

        if limit > 0:
            mix_fpath_list = mix_fpath_list[:limit]
            s1_fpath_list = s1_fpath_list[:limit]
            s2_fpath_list = s2_fpath_list[:limit]

        print(f"Founds {len(mix_fpath_list)} files")

        self.mix_fpath_list = mix_fpath_list
        self.s1_fpath_list = s1_fpath_list
        self.s2_fpath_list = s2_fpath_list
        self.sr = sr
        self.duration = duration
        self.sample_length = int(sr * duration)
        self.is_train = is_train

    def __len__(self):
        return len(self.mix_fpath_list)

    def __getitem__(self, index):
        stem = Path(self.mix_fpath_list[index]).stem
        mix_fpath = self.mix_fpath_list[index]
        s1_fpath = self.s1_fpath_list[index]
        s2_fpath = self.s2_fpath_list[index]
        mix_y, _ = sf.read(mix_fpath)
        s1_y, _ = sf.read(s1_fpath)
        s2_y, _ = sf.read(s2_fpath)

        if self.is_train:
            mix_y, start_idx = subsample(mix_y, self.sample_length, return_start_idx=True)
            s1_y = subsample(s1_y, self.sample_length, start_idx=start_idx)
            s2_y = subsample(s2_y, self.sample_length, start_idx=start_idx)

        mix_y = mix_y.astype(np.float32)
        s1_y = s1_y.astype(np.float32)
        s2_y = s2_y.astype(np.float32)
        ref_y = np.stack([s1_y, s2_y], axis=0)

        return mix_y, ref_y, stem
