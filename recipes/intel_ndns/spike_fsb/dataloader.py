import glob
import os
import re

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from audiozen.acoustics.audio_feature import subsample


class DNSAudio(Dataset):
    def __init__(self, root="./", limit=None, offset=0, train=True) -> None:
        """Audio dataset loader for DNS.

        Args:
            root: Path of the dataset location, by default './'.
        """
        super().__init__()
        self.root = root
        print(f"Loading dataset from {root}...")
        self.noisy_files = glob.glob(root + "noisy/**.wav")

        if offset > 0:
            self.noisy_files = self.noisy_files[offset:]

        if limit:
            self.noisy_files = self.noisy_files[:limit]

        self.file_id_from_name = re.compile("fileid_(\d+)")
        self.snr_from_name = re.compile("snr(-?\d+)")
        self.target_level_from_name = re.compile("tl(-?\d+)")
        self.source_info_from_name = re.compile("^(.*?)_snr")

        self.train = train

    def _get_filenames(self, n):
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = self.root + f"clean/clean_fileid_{file_id}.wav"
        noise_file = self.root + f"noise/noise_fileid_{file_id}.wav"
        snr = int(self.snr_from_name.findall(filename)[0])
        target_level = int(self.target_level_from_name.findall(filename)[0])
        source_info = self.source_info_from_name.findall(filename)[0]
        metadata = {
            "snr": snr,
            "target_level": target_level,
            "source_info": source_info,
        }
        return noisy_file, clean_file, noise_file, metadata

    def __getitem__(self, n):
        """Gets the nth sample from the dataset.

        Args:
            n: Index of the sample to be retrieved.

        Returns:
            Noisy audio sample, clean audio sample, noise audio sample, sample metadata.
        """
        noisy_file, clean_file, noise_file, metadata = self._get_filenames(n)
        noisy_audio, sampling_frequency = sf.read(noisy_file)
        clean_audio, _ = sf.read(clean_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
        train_num_samples = 6 * sampling_frequency
        metadata["fs"] = sampling_frequency

        if len(noisy_audio) > num_samples:
            noisy_audio = noisy_audio[:num_samples]
        else:
            noisy_audio = np.concatenate(
                [noisy_audio, np.zeros(num_samples - len(noisy_audio))]
            )
        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate(
                [clean_audio, np.zeros(num_samples - len(clean_audio))]
            )

        noisy_audio = noisy_audio.astype(np.float32)
        clean_audio = clean_audio.astype(np.float32)

        if self.train:
            noisy_audio, start_position = subsample(
                noisy_audio,
                sub_sample_length=train_num_samples,
                return_start_position=True,
            )
            clean_audio = subsample(
                clean_audio,
                sub_sample_length=train_num_samples,
                start_position=start_position,
            )

        return noisy_audio, clean_audio, noisy_file

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.noisy_files)


if __name__ == "__main__":
    train_set = DNSAudio(root="../../data/MicrosoftDNS_4_ICASSP/training_set/")
    validation_set = DNSAudio(root="../../data/MicrosoftDNS_4_ICASSP/validation_set/")
