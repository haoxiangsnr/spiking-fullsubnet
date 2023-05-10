from pathlib import Path
from random import shuffle

import librosa

config = {
    "root": "",
    "fs": 16000,
    "audio_length": 30,
    "silence_length": 0.2,
    "total_hours": 500,
    "snr_lower": -5,
    "snr_upper": 20,
    "target_level_lower": -35,
    "target_snr_levels": 21,
    "clean_activity_threshold": 0.6,
    "noise_activity_threshold": 0.0,
    "is_validation": False,
    "noisy_speech_dir": "",
    "clean_speech_dir": "",
    "noise_dir": "",
    "log_dir": "",
    "file_index_start": 0,
}


def segmental_snr_mixer():
    pass


def main(config):
    data_root_dir = Path(config["root"]).expanduser().absolute()
    clean_dir = data_root_dir / "clean_fullband"
    noise_dir = data_root_dir / "noise_fullband"

    clean_file_path_list = librosa.util.find_files(clean_dir)
    noise_file_path_list = librosa.util.find_files(noise_dir)

    shuffle(clean_file_path_list)
    shuffle(noise_file_path_list)
