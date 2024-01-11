from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm


try:
    import pyloudnorm as pyln
except ImportError:
    print("Please install pyloudnorm first.")
    print("pip install pyloudnorm")
    exit()


def normalize_to_noisy(src_fpath, ref_fpath, src_dir, dest_dir):
    if isinstance(src_fpath, str):
        src_fpath = Path(src_fpath)
    if isinstance(ref_fpath, str):
        ref_fpath = Path(ref_fpath)

    assert src_fpath.stem == ref_fpath.stem, f"{src_fpath} and {ref_fpath} have different name."

    dest_fpath = dest_dir / src_fpath.relative_to(src_dir)
    dest_fpath.parent.mkdir(parents=True, exist_ok=True)

    # load noisy audio
    y, sr = librosa.load(src_fpath, sr=48000)
    # load reference audio
    y_ref, sr_ref = librosa.load(ref_fpath, sr=48000)
    assert sr == sr_ref, f"{src_fpath} and {ref_fpath} have different sampling rate."

    # normalize to reference audio
    meter = pyln.Meter(sr)
    y_loudness = meter.integrated_loudness(y)
    y_ref_loudness = meter.integrated_loudness(y_ref)

    y = pyln.normalize.loudness(y, y_loudness, y_ref_loudness)

    # if clipping, then max normalize
    if np.max(np.abs(y)) > 1.0:
        y = y / np.max(np.abs(y))

    sf.write(dest_fpath.as_posix(), y, sr)

    return y_loudness, y_ref_loudness


def main(src_dir, ref_dir, dest_dir, num_workers):
    # audio files which needed to be normalized
    src_dir = Path(src_dir).expanduser().absolute()
    # audio files which used as reference.
    ref_dir = Path(ref_dir).expanduser().absolute()
    # destination directory
    dest_dir = Path(dest_dir).expanduser().absolute()

    src_fpath_list = librosa.util.find_files(src_dir.as_posix(), ext=["wav"])
    ref_fpath_list = librosa.util.find_files(ref_dir.as_posix(), ext=["wav"])

    loudness = Parallel(n_jobs=num_workers)(
        delayed(normalize_to_noisy)(src_fpath, ref_fpath, src_dir, dest_dir)
        for src_fpath, ref_fpath in tqdm(zip(src_fpath_list, ref_fpath_list))
    )

    rows = []
    for src_fpath, ref_fpath, (y_loudness, y_ref_loudness) in zip(src_fpath_list, ref_fpath_list, loudness):
        rows.append(
            {
                "noisy": y_ref_loudness,
                "enhanced": y_loudness,
                "src": src_fpath,
                "ref": ref_fpath,
            }
        )
    pd.DataFrame(rows).to_csv(dest_dir / "loudness.csv", index=False)
    print(f"Loudness normalization finished. (dest_dir: {dest_dir / 'loudness.csv'})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=20)
    args = parser.parse_args()
    main(**vars(args))
