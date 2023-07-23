import glob
import os
import random
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from audiozen.acoustics.audio_feature import subsample
from audiozen.dataset.base_dataset import BaseDataset


class DNSAudio(BaseDataset):
    def __init__(self, root_dir, limit=None, offset=0, sub_sample_len=-1):
        """Audio dataset loader for DNS.

        Args:
            root: Path of the dataset location, by default './'.
        """
        super().__init__()
        self.root_dir = Path(root_dir).expanduser().absolute()
        self.noisy_files = librosa.util.find_files(
            (self.root_dir / "noisy").as_posix(), ext="wav"
        )
        print(f"Found {len(self.noisy_files)} files in {root_dir}noisy")

        if offset > 0:
            self.noisy_files = self.noisy_files[offset:]

        if limit:
            self.noisy_files = self.noisy_files[:limit]

        self.file_id_from_name = re.compile("fileid_(\d+)")
        self.snr_from_name = re.compile("snr(-?\d+)")
        self.target_level_from_name = re.compile("tl(-?\d+)")
        self.source_info_from_name = re.compile("^(.*?)_snr")

        self.sub_sample_len = sub_sample_len

    def __len__(self) -> int:
        return len(self.noisy_files)

    def _find_clean_fpath(self, noisy_fpath):
        filename = noisy_fpath.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_fpath = (
            self.root_dir / "clean" / f"clean_fileid_{file_id}.wav"
        ).as_posix()

        return clean_fpath

    @staticmethod
    def _load_wav_offset(
        path, duration=None, sr=None, mode="wrap", offset=None, return_offset=False
    ):
        if isinstance(path, Path):
            path = path.as_posix()

        offset = 0
        with sf.SoundFile(path) as sf_desc:
            orig_sr = sf_desc.samplerate

            if duration is not None:
                frame_orig_duration = sf_desc.frames
                frame_duration = int(duration * orig_sr)
                if frame_duration < frame_orig_duration:
                    # Randomly select a segment
                    if offset is None:
                        offset = np.random.randint(frame_orig_duration - frame_duration)

                    sf_desc.seek(offset)
                    y = sf_desc.read(
                        frames=frame_duration, dtype=np.float32, always_2d=True
                    ).T
                else:
                    y = sf_desc.read(dtype=np.float32, always_2d=True).T  # [C, T]
                    y = np.pad(
                        y, ((0, 0), (0, frame_duration - frame_orig_duration)), mode
                    )
            else:
                y = sf_desc.read(dtype=np.float32, always_2d=True).T

        if y.shape[0] == 1:
            y = y[0]

        if sr is not None:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        else:
            sr = orig_sr

        if return_offset:
            return y, sr, offset
        else:
            return y, sr

    def __getitem__(self, _):
        """Gets the nth sample from the dataset.

        Args:
            n: Index of the sample to be retrieved.

        Returns:
            Noisy audio sample, clean audio sample, noise audio sample, sample metadata.
        """
        noisy_fpath = random.choice(self.noisy_files)
        clean_fpath = self._find_clean_fpath(noisy_fpath)

        if self.sub_sample_len > 0:
            # Use `offset` and `duration` arguments
            # For the training set, we want to load a segment with a fixed length
            # We also want to load the same start position for the clean and noisy files
            # Here we use the offset returned by the noisy file to load the clean file
            noisy_y, _, offset = self._load_wav_offset(
                noisy_fpath,
                duration=self.sub_sample_len,
                sr=16000,
                offset=None,
                return_offset=True,
                mode="constant",  # very few are shorter than 30s
            )
            clean_y, _, _ = self._load_wav_offset(
                clean_fpath,
                duration=self.sub_sample_len,
                sr=16000,
                offset=offset,
                mode="constant",
                return_offset=True,
            )
        else:
            # Only use `duration` argument
            # For the validation set, we want to load the entire file
            # Some files are shorter than 30s, so we pad them with zeros
            noisy_y, _ = self._load_wav_offset(
                noisy_fpath,
                sr=16000,
                duration=30,
                mode="constant",
            )
            clean_y, _ = self._load_wav_offset(
                clean_fpath,
                sr=16000,
                duration=30,
                mode="constant",
            )

        return noisy_y, clean_y


if __name__ == "__main__":
    train_set = DNSAudio(
        root_dir="/datasets/datasets_fullband/training_set",
        sub_sample_len=6,
    )
    print(len(train_set))
    for i in range(10):
        print(train_set[i][0].shape)
        print(train_set[i][1].shape)

    valid_set = DNSAudio(
        root_dir="/datasets/datasets_fullband/validation_set",
        limit=10,
    )
    print(len(valid_set))
    for i in range(len(valid_set)):
        print(valid_set[i][0].shape)
        print(valid_set[i][1].shape)
