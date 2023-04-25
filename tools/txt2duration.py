"""
Check the total duration of a list of wav files from a text file.

Usage:
    python txt2duration.py <text_path>
"""
from pathlib import Path

import librosa
from tqdm import tqdm


def main(text_path):
    with open(text_path, "r") as f:
        wav_paths = f.readlines()

    wav_paths = [Path(p.strip()) for p in tqdm(wav_paths)]

    total_duration = 0
    for wav_path in wav_paths:
        dur = librosa.get_duration(filename=wav_path)
        total_duration += dur

    # get min and seconds first
    mm, ss = divmod(total_duration, 60)
    # Get hours
    hh, mm = divmod(mm, 60)

    print("Time in Seconds:", hh, "Hours", mm, "Minutes", ss, "Seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text_path", type=str)
    args = parser.parse_args()

    main(args.text_path)
