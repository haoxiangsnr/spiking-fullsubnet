import argparse
from pathlib import Path

import numpy as np


def find_enrollment(target_path):
    path = Path(str(target_path).replace("seg/clean", "enrollment_wav"))
    new_stem = "_".join(path.stem.split("_")[:-1])
    filename = f"enrol_{new_stem}.wav"  # e.g., enrol_complete_french_mix_sous_les_mers_1_01_seg1
    enroll_path = Path(path).parent / filename

    return enroll_path


def find_embedding(target_path):
    # /path/to/complete_french_mix_sous_les_mers_1_01_seg1.wav
    path = Path(str(target_path).replace("seg/clean", "enrollment_embedding_ecapa"))

    # complete_french_mix_sous_les_mers_1_01.wav
    new_stem = "_".join(path.stem.split("_")[:-1])

    # enrol_complete_french_mix_sous_les_mers_1_01.npy
    filename = f"enrol_{new_stem}.npy"

    embedding_path = Path(path).parent / filename

    return embedding_path


def main(clean_fpath_file, output_file):
    clean_list = np.loadtxt(clean_fpath_file, dtype=str)

    print(f"total {len(clean_list)} clean files")

    missing_num = 0
    paired_file_path_list = []
    for clean_path in clean_list:
        enroll_path = find_enrollment(clean_path)
        embedding_path = find_embedding(clean_path)

        if enroll_path.exists() and embedding_path.exists():
            clean_path = Path(clean_path).expanduser().absolute()
            paired_file_path_list.append(clean_path.as_posix())
        else:
            missing_num += 1
            print(f"enrollment wav not found: {enroll_path}")

    print(f"missing {missing_num} files out of {len(clean_list)}")

    with open(output_file, "w") as f:
        for path in paired_file_path_list:
            f.write(f"{path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_fpath_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
