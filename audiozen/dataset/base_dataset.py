from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from torch.utils import data
from tqdm import tqdm

from audiozen.acoustics.audio_feature import load_wav


class BaseDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _load_dataset_from_text_and_dir_list(dataset_list):
        """Load dataset from text file or directory.

        Args:
            dataset_list: A list of dataset path, which can be a directory or a text file. `dataset_list` can also be a single path of directory or text file.

        Raises:
            FileNotFoundError: If the dataset path is not found.

        Returns:
            A list of file path.
        """
        if isinstance(dataset_list, str):
            dataset_list = [dataset_list]

        fpath_list = []

        for dataset in dataset_list:
            dataset = Path(dataset).expanduser().absolute()
            if dataset.is_dir():
                fpath_list += librosa.util.find_files(dataset.as_posix(), ext="wav")
            elif dataset.is_file():
                with open(dataset, "r") as f:
                    fpath_list += [line.rstrip("\n") for line in f]
            else:
                raise FileNotFoundError(f"File {dataset} not found.")

        return fpath_list

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
        assert len(snr_range) == 2, f"The range of SNR should be [low, high], not {snr_range}."
        assert snr_range[0] <= snr_range[-1], "The low SNR should not larger than high SNR."

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)

        return snr_list

    def _preload_dataset(self, file_path_list, remark=""):
        waveform_list = Parallel(n_jobs=self.num_workers)(
            delayed(self._load_wav)(f_path, sr=self.sr) for f_path in tqdm(file_path_list, desc=remark)
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

    @staticmethod
    def _load_wav(path, duration=None, sr=None, mode="wrap"):
        if isinstance(path, Path):
            path = path.as_posix()

        with sf.SoundFile(path) as sf_desc:
            orig_sr = sf_desc.samplerate

            if duration is not None:
                frame_orig_duration = sf_desc.frames
                frame_duration = int(duration * orig_sr)
                if frame_duration < frame_orig_duration:
                    # Randomly select a segment
                    offset = np.random.randint(frame_orig_duration - frame_duration)
                    sf_desc.seek(offset)
                    y = sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=True).T
                else:
                    y = sf_desc.read(dtype=np.float32, always_2d=True).T  # [C, T]
                    y = np.pad(y, ((0, 0), (0, frame_duration - frame_orig_duration)), mode)
            else:
                y = sf_desc.read(dtype=np.float32, always_2d=True).T

        if y.shape[0] == 1:
            y = y[0]

        if sr is not None:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        else:
            sr = orig_sr

        return y, sr
