"""
Reading a file and checking if the wav paths in this file exists or not.

Usage:
    python tools/check_existence.py <path_to_file>
"""

import argparse
from pathlib import Path

from tqdm import tqdm


def main(fpath):
    fpath = Path(fpath).expanduser().resolve()
    with open(fpath, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.exists():
                print(f"Path {path} does not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", help="Path to the file")
    args = parser.parse_args()

    main(args.fpath)
