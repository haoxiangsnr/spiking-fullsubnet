from pathlib import Path

import h5py
import librosa
import numpy as np
from tqdm import tqdm

noise_path = "/datasets/datasets_fullband/training_set/noise"
hdf5_path = "/datasets/datasets_fullband/training_set/noise_fp16.hdf5"

# Load the audio file
noise_fpath_list = librosa.util.find_files(noise_path, ext=["wav"])

# Save the audio file to HDF5
# with h5py.File(hdf5_path, "w") as hf:
#     for noise_fpath in tqdm(noise_fpath_list):
#         noise, sr = librosa.load(noise_fpath, sr=16000)
#         noise = noise.astype(np.float16)

#         noise_file_stem = Path(noise_fpath).stem

#         hf.create_dataset(noise_file_stem, data=noise)

# Save the audio file to HDF5 using audio file stem as key
noise_data = []
noise_file_stem_list = []
for noise_fpath in tqdm(noise_fpath_list):
    noise, sr = librosa.load(noise_fpath, sr=16000)
    noise = noise.astype(np.float16)

    noise_data.append(noise)
    noise_file_stem_list.append(Path(noise_fpath).stem)

# Save the various lengths of noise data to HDF5
