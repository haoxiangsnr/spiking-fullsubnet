from pathlib import Path

import click
import librosa
from tqdm import tqdm


@click.command()
@click.argument("root_dir", type=click.Path(exists=True))
def main(root_dir):
    root_dir = Path(root_dir).expanduser().resolve()

    wav_fpath_list = librosa.util.find_files(root_dir, ext="wav")

    num_non_mono_wav = 0

    # Check the channel number of each audio file
    for wav_fpath in tqdm(wav_fpath_list):
        wav, sr = librosa.load(wav_fpath, sr=None)
        if wav.ndim != 1:
            print(f"{wav_fpath} has {wav.ndim} channels")
            num_non_mono_wav += 1

    print(f"Number of non-mono wav files: {num_non_mono_wav}")


if __name__ == "__main__":
    main()
