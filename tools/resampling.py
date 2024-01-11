"""
Resampling a nested directory of wav files to a target sampling rate while keeping the directory structure.

Dependencies:
    pip install librosa soundfile joblib tqdm

Note:
    This script will omit symbolic links. Please check the source directory before running this script.

Usage:
    python resampling.py \
        --src_dir /path/to/LibriSpeech/train-clean-100 \
        --dest_dir /path/to/LibriSpeech/train-clean-100-16k \
        --target_sr 16000 \
        --num_workers 40
"""
import argparse
from pathlib import Path

import librosa
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm


def resampling(fpath, src_dir, dest_dir, target_sr):
    if isinstance(fpath, str):
        fpath = Path(fpath)

    dest_fpath = dest_dir / fpath.relative_to(src_dir)
    dest_fpath.parent.mkdir(parents=True, exist_ok=True)

    y, _ = librosa.load(fpath, sr=target_sr)
    sf.write(dest_fpath.as_posix(), y, target_sr)


def main(src_dir, dest_dir, target_sr, num_workers, ext):
    src_dir = Path(src_dir).expanduser().absolute()
    dest_dir = Path(dest_dir).expanduser().absolute()

    fpath_list = librosa.util.find_files(src_dir.as_posix(), ext=ext, recurse=True)

    Parallel(n_jobs=num_workers)(
        delayed(resampling)(fpath, src_dir, dest_dir, target_sr) for fpath in tqdm(fpath_list)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="source directory")
    parser.add_argument("--dest_dir", type=str, required=True, help="destination directory")
    parser.add_argument("--target_sr", type=int, default=16000, required=True, help="target sampling rate")
    parser.add_argument("--num_workers", type=int, default=40, required=True, help="number of workers")
    parser.add_argument("--ext", type=str, nargs="+", default=["wav"], help="file extensions")
    args = parser.parse_args()
    main(**vars(args))
