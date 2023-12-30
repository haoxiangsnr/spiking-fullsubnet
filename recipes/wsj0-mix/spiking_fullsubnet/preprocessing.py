from pathlib import Path

import librosa
from tqdm import tqdm

accepted_duration = 4
sr = 8000

mix_dir = Path("/datasets/wsj0-mix/2speakers/wav8k/min/tr/mix")
s1_dir = Path("/datasets/wsj0-mix/2speakers/wav8k/min/tr/s1")
s2_dir = Path("/datasets/wsj0-mix/2speakers/wav8k/min/tr/s2")
mix_scp_fpath = f"/home/xhao/proj/spiking-fullsubnet/recipes/wsj0-mix/spiking_fullsubnet/min_tr_mix.scp"
s1_scp_fpath = f"/home/xhao/proj/spiking-fullsubnet/recipes/wsj0-mix/spiking_fullsubnet/min_tr_s1.scp"
s2_scp_fpath = f"/home/xhao/proj/spiking-fullsubnet/recipes/wsj0-mix/spiking_fullsubnet/min_tr_s2.scp"

# Make sure the length of audio files are acceptable
accepted_sample_length = int(sr * accepted_duration)
print(f"Checking acceptable length in {mix_dir}...")

mix_output_list = []
mix_fpath_list = librosa.util.find_files(mix_dir)
print(f"Number of audio files: {len(mix_fpath_list)}")

for fpath in tqdm(mix_fpath_list):
    duration = librosa.get_duration(path=fpath, sr=sr)
    if duration < accepted_duration:
        pass
    else:
        mix_output_list.append(fpath)
print(f"Number of acceptable audio files: {len(mix_output_list)}")

# Save the list of acceptable audio files
print(f"Saving the list of acceptable audio files to {mix_scp_fpath}...")
with open(mix_scp_fpath, "w") as f:
    for fpath in mix_output_list:
        f.write(f"{fpath}\n")

# Copy the list of acceptable audio files to s1_dir and s2_dir
print(f"Copying the list of acceptable audio files to {s1_dir} and {s2_dir}...")
s1_output_list = []
s2_output_list = []
for fpath in mix_output_list:
    stem = Path(fpath).stem
    s1_fpath = s1_dir / f"{stem}.wav"
    s2_fpath = s2_dir / f"{stem}.wav"

    # Assert that s1_fpath and s2_fpath exist
    assert s1_fpath.exists()
    assert s2_fpath.exists()

    s1_output_list.append(s1_fpath)
    s2_output_list.append(s2_fpath)

# Save the list of acceptable audio files
with open(s1_scp_fpath, "w") as f:
    for fpath in s1_output_list:
        f.write(f"{fpath}\n")

with open(s2_scp_fpath, "w") as f:
    for fpath in s2_output_list:
        f.write(f"{fpath}\n")
