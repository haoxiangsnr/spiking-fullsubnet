import glob
import os
import random
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from audiozen.dataset.base_dataset import BaseDataset


class DNSAudio(BaseDataset):
    def __init__(self, root, limit=None, offset=0, sub_sample_len=-1):
        """Audio dataset loader for DNS.

        Args:
            root: Path of the dataset location, by default './'.
        """
        super().__init__()
        self.root_dir = Path(root).expanduser().absolute()
        self.noisy_dir = self.root_dir / "noisy"
        self.clean_dir = self.root_dir / "clean"

        # Find all the noisy files
        self.noisy_files = glob.glob(f"{self.noisy_dir}/**.wav")
        print(f"Found {len(self.noisy_files)} files in {root}")

        if offset > 0:
            self.noisy_files = self.noisy_files[offset:]

        if limit:
            self.noisy_files = self.noisy_files[:limit]

        self.file_id_from_name = re.compile("fileid_(\d+)")
        self.snr_from_name = re.compile("snr(-?\d+)")
        self.target_level_from_name = re.compile("tl(-?\d+)")
        self.source_info_from_name = re.compile("^(.*?)_snr")

        self.sub_sample_len = sub_sample_len
        self.length = len(self.noisy_files)

    def __len__(self) -> int:
        return self.length

    def _find_clean_fpath(self, noisy_fpath):
        filename = noisy_fpath.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_fpath = self.clean_dir / f"clean_fileid_{file_id}.wav"
        return clean_fpath

    @staticmethod
    def _load_wav_offset(
        path,
        duration=None,
        sr=None,
        mode="constant",
        offset=None,
        return_offset=False,
    ):
        """Loads a wav file and returns a numpy array.

        Args:
            path: a string or a Path object.
            duration: seconds to load. If None, load the entire file.
            sr: sample rate. If None, use the original sample rate.
            mode: _description_. Defaults to "constant".
            offset: only works when duration is not None.

        Returns:
            _description_
        """
        if isinstance(path, Path):
            path = path.as_posix()

        with sf.SoundFile(path) as sf_desc:
            orig_sr = sf_desc.samplerate

            if duration is None:
                # Load the entire file with no slicing and padding
                y = sf_desc.read(dtype=np.float32, always_2d=True).T
            else:
                # Load a segment of the file with slicing or padding
                frame_orig_duration = sf_desc.frames
                frame_duration = int(duration * orig_sr)

                # If the desired duration is shorter than the original duration, we need to slice
                if frame_duration < frame_orig_duration:
                    if offset is None:
                        offset = np.random.randint(frame_orig_duration - frame_duration)

                    sf_desc.seek(offset)

                    y = sf_desc.read(
                        frames=frame_duration, dtype=np.float32, always_2d=True
                    ).T
                # If the desired duration is longer than the original duration, we need to pad
                else:
                    y = sf_desc.read(dtype=np.float32, always_2d=True).T  # [C, T]
                    y = np.pad(
                        y, ((0, 0), (0, frame_duration - frame_orig_duration)), mode
                    )

        if y.shape[0] == 1:
            y = y[0]

        if sr is None:
            sr = orig_sr
        else:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

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
        sr = 16000

        if self.sub_sample_len > 0:
            # Use `offset` and `duration` arguments
            # For the training set, we want to load a segment with a fixed length
            # We also want to load the same start position for the clean and noisy files
            # Here we use the offset returned by the noisy file to load the clean file
            noisy_y, _, offset = self._load_wav_offset(
                noisy_fpath,
                duration=self.sub_sample_len,
                sr=sr,
                offset=None,
                return_offset=True,
                mode="constant",  # very few are shorter than 30s
            )
            clean_y, _, _ = self._load_wav_offset(
                clean_fpath,
                duration=self.sub_sample_len,
                sr=sr,
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
                sr=sr,
                duration=30,
                mode="constant",
            )
            clean_y, _ = self._load_wav_offset(
                clean_fpath,
                sr=sr,
                duration=30,
                mode="constant",
            )

        return noisy_y, clean_y, "placeholder"


if __name__ == "__main__":
    import soundfile as sf

    train_set = DNSAudio(
        root="/datasets/datasets_fullband/training_set",
        sub_sample_len=-1,
    )
    for i in range(10):
        noisy, clean, _ = train_set[i]
        print(noisy.shape, clean.shape)
        sf.write(f"tmp/noisy_{i}.wav", noisy, 16000)
        sf.write(f"tmp/clean_{i}.wav", clean, 16000)
