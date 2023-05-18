from audiozen.acoustics.audio_feature import find_files, load_wav
from audiozen.dataset.base_dataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, path_or_path_list, sr=16000, limit=None, offset=0):
        super().__init__()
        self.file_paths = find_files(path_or_path_list, limit=limit, offset=offset)
        self.sr = sr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        wav = load_wav(file_path, self.sr)
        return wav, file_path
