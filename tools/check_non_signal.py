from pathlib import Path

import click
import librosa
from tqdm.auto import tqdm


@click.command()
@click.argument("root_dir", type=click.Path(exists=True))
@click.option(
    "threshold",
    "--threshold",
    type=float,
    default=0.1,
    help="Recommend 0.1 to avoid removing short RIRs.",
)
def main(root_dir, threshold):
    fpath_list = librosa.util.find_files(root_dir, ext="wav")
    print(f"Founds {len(fpath_list)} wav files in {root_dir}")

    for fpath in tqdm(fpath_list):
        dur = librosa.get_duration(path=fpath)
        if dur < threshold:
            print(f"{fpath} is too short ({dur} sec)")
            # remove the file
            Path(fpath).unlink()
            # print the command to remove the file
            print(f"rm -rf {fpath}")


if __name__ == "__main__":
    main()
