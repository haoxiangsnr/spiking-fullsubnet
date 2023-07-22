from pathlib import Path

import click
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from audiozen.acoustics.audio_feature import activity_detector


@click.command()
@click.argument("src_dir", type=click.Path(exists=True))
@click.argument("dest_dir", type=click.Path())
@click.option(
    "clip_threshold",
    "--clip_threshold",
    type=float,
    default=0.99,
    help="avoid clipping",
)
@click.option(
    "activity_threshold",
    "--activity_threshold",
    type=float,
    default=0.6,
    help="keep high activity",
)
def main(src_dir, dest_dir, clip_threshold, activity_threshold):
    src_dir = Path(src_dir).expanduser().resolve()
    dest_dir = Path(dest_dir).expanduser().resolve()

    wav_fpath_list = librosa.util.find_files(src_dir, ext="wav")
    num_valid_wav = 0
    clipped_wav_fpath = []
    inactive_wav_fpath = []

    for wav_fpath in tqdm(wav_fpath_list):
        wav, sr = librosa.load(wav_fpath, sr=None)
        if np.max(np.abs(wav)) > clip_threshold:
            print(f"{wav_fpath} is clipped")
            clipped_wav_fpath.append(wav_fpath)
            continue

        perc_active = activity_detector(wav)
        if perc_active < activity_threshold:
            print(f"{wav_fpath} is not active ({perc_active})")
            inactive_wav_fpath.append(wav_fpath)
            continue

        wav_fpath = Path(wav_fpath)
        dest_fpath = dest_dir / wav_fpath.relative_to(src_dir)
        dest_fpath.parent.mkdir(parents=True, exist_ok=True)

        sf.write(dest_fpath.as_posix(), wav, sr)
        num_valid_wav += 1

    print(f"Valid wav files: {num_valid_wav} out of {len(wav_fpath_list)} in {src_dir}")
    print(f"After checking clipping...")
    print(f"Clipped wav files: {len(clipped_wav_fpath)}")
    print(f"After checking inactive wav files...")
    print(f"Inactive wav files: {len(inactive_wav_fpath)}")
    print(f"Total: {len(clipped_wav_fpath) + len(inactive_wav_fpath)}")


if __name__ == "__main__":
    main()
