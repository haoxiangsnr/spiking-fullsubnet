import random
from pathlib import Path

import librosa


def main(src_dirs, limit=None, output_txt="filelist.txt"):
    src_dirs = [Path(src_dir).resolve() for src_dir in src_dirs]
    output_txt = Path(output_txt).resolve()
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    fpath_list = []
    for src_dir in src_dirs:
        print(f"Searching {src_dir.as_posix()}...")
        fpath_list += librosa.util.find_files(src_dir.as_posix(), ext=["wav"])

    orig_len = len(fpath_list)
    print(f"Found {orig_len} files in {src_dir.as_posix()}.")

    random.shuffle(fpath_list)

    if limit is not None:
        fpath_list = fpath_list[:limit]

    print(f"Saving {len(fpath_list)} files to {output_txt.as_posix()}.")
    with open(output_txt.as_posix(), "w") as f:
        for fpath in fpath_list:
            f.write(fpath + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dirs", type=str, nargs="+", required=True)
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument(
        "-o", "--output_txt", type=str, required=True, default="filelist.txt"
    )
    args = parser.parse_args()

    main(args.src_dirs, args.num, args.output_txt)
