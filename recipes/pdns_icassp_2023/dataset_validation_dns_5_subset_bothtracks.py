from pathlib import Path

import librosa
import numpy as np

from audiozen.acoustics.audio_feature import load_wav
from audiozen.dataset.base_dataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(
        self,
        noisy_dir,
        sr,
        offset=0,
        limit=None,
        return_enrollment=True,
        return_embedding=False,
    ):
        super().__init__()
        noisy_files_list = []
        noisy_dir = Path(noisy_dir).expanduser().absolute()
        noisy_files_list += librosa.util.find_files(noisy_dir.as_posix())

        if offset > 0:
            noisy_files_list = noisy_files_list[offset:]

        if limit:
            noisy_files_list = noisy_files_list[:limit]

        self.length = len(noisy_files_list)
        self.noisy_files_list = noisy_files_list
        self.sr = sr
        self.return_enrollment = return_enrollment
        self.return_embedding = return_embedding

    def __len__(self):
        return self.length

    def _find_enrollment_path(self, noisy_path):
        enroll_path = Path(str(noisy_path).replace("noisy_subset", "enrol"))

        if not enroll_path.exists():
            raise FileNotFoundError(enroll_path.as_posix())

        return enroll_path.as_posix()

    def _find_embedding_path(self, noisy_path):
        embed_path = Path(str(noisy_path).replace("noisy_subset", "enrol_embedding"))
        embed_path = embed_path.with_suffix(".npy")

        if not embed_path.exists():
            raise FileNotFoundError(embed_path.as_posix())

        return embed_path.as_posix()

    def __getitem__(self, item):
        noisy_path = self.noisy_files_list[item]
        noisy_y = load_wav(noisy_path, sr=self.sr)
        output = [noisy_y, noisy_path]

        if self.return_enrollment:
            enroll_path = self._find_enrollment_path(noisy_path)
            enroll_y = load_wav(enroll_path, sr=self.sr)
            output.append(enroll_y)

        if self.return_embedding:
            embed_path = self._find_embedding_path(noisy_path)
            embed_y = np.load(embed_path)
            output.append(embed_y)

        return output
