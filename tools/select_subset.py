import random
import shutil
from pathlib import Path

import librosa


def main(src_dir, dest_dir, num, len_threshold):
    src_dir = Path(src_dir).expanduser().absolute()
    dest_dir = Path(dest_dir).expanduser().absolute()

    fpath_list = librosa.util.find_files(src_dir.as_posix(), ext=["wav"])
    orig_len = len(fpath_list)
    print(f"Found {orig_len} files in {src_dir.as_posix()}.")

    random.shuffle(fpath_list)
    subset_fpath_list = fpath_list[:num]
    print(f"Selected {len(subset_fpath_list)} files.")

    i = 0
    for fpath in subset_fpath_list:
        duration = librosa.get_duration(filename=fpath)

        if duration > len_threshold:
            continue

        fpath = Path(fpath)
        dest_fpath = dest_dir / fpath.relative_to(src_dir)
        dest_fpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fpath, dest_fpath)
        i += 1

    print(f"Copied {i} files to {dest_dir.as_posix()}.")
    print(f"Skipped {len(subset_fpath_list) - i} files.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", type=str, required=True)
    parser.add_argument("-o", "--dest_dir", type=str, required=True)
    parser.add_argument("-n", "--num", type=int, required=True)
    parser.add_argument("--len_threshold", type=float, default=30)
    args = parser.parse_args()

    main(args.src_dir, args.dest_dir, args.num, args.len_threshold)
