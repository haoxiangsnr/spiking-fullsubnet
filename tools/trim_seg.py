import argparse
from pathlib import Path

import librosa
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm


def trim_and_segment(src_fpath, src_dir, dest_dir, sr, min_duration, max_duration):
    """Split audio files into segments of fixed seconds.

    Steps:
        1. Load audio file with target sample rate.
        2. Trim silence from the beginning and end of the audio file.
        3. Split audio into segments of 'max_duration' seconds and save them.

    Note:
        If the audio file is shorter than 'max_duration' seconds and longer than 'min_duration' seconds, save the audio file as a segment.
        If the audio file is longer than 'max_duration' seconds, split the audio file into segments.

    Args:
        src_fpath: Path to the source audio file.
        src_dir: Path to the source root directory.
        dest_dir: Path to the destination root directory.
        sr: Target sample rate of the audio files.
        min_duration: Lower bound of the duration of the audio files.
        max_duration: Duration of each segment.
    """
    if isinstance(src_fpath, str):
        src_fpath = Path(src_fpath)

    dest_fpath = dest_dir / src_fpath.relative_to(src_dir)
    dest_fpath.parent.mkdir(parents=True, exist_ok=True)

    # Load audio file with target sample rate
    try:
        y, _ = librosa.load(src_fpath, sr=sr)
    except ValueError:
        print(f"File {src_fpath} is corrupted.")
        return
    except:
        print(f"File {src_fpath} is not a valid audio file.")
        return

    print(f"Length of audio file {src_fpath}: {len(y) / sr:.2f} seconds.")

    # Trim silence from the beginning and end of the audio file
    audio, index = librosa.effects.trim(y)

    if len(audio) < int(min_duration * sr):
        print(f"Audio file {src_fpath} is too short.")
        return

    # Split audio into segments of 'max_duration' seconds and save them
    len_seg_sample = int(max_duration * sr)
    if len(audio) > len_seg_sample:
        for idx, offset in enumerate(range(0, len(audio), len_seg_sample)):
            seg_tag = src_fpath.stem + "_seg" + str(idx + 1)
            seg_fpath = dest_fpath.parent / (seg_tag + ".wav")
            seg = audio[offset : offset + len_seg_sample]
            # After splitting, some segments may be shorter than 4 seconds.
            if len(seg) < int(min_duration * sr):
                print(f"Segment {seg_fpath} is too short.")
                return
            sf.write(seg_fpath.as_posix(), seg, sr)
    else:
        sf.write(dest_fpath.as_posix(), audio, sr)


def main(src_dir, dest_dir, sr, min_duration, max_duration, num_workers):
    src_dir = Path(src_dir).expanduser().absolute()
    dest_dir = Path(dest_dir).expanduser().absolute()

    file_paths = librosa.util.find_files(src_dir.as_posix(), ext=["wav"])
    print(f"Found {len(file_paths)} audio files in {src_dir}.")

    Parallel(n_jobs=num_workers)(
        delayed(trim_and_segment)(
            file_path,
            src_dir,
            dest_dir,
            sr=sr,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        for file_path in tqdm(file_paths)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim and segment audio files.")
    parser.add_argument(
        "-i",
        "--src_dir",
        type=str,
        required=True,
        help="Path to the source directory.",
    )
    parser.add_argument(
        "-o",
        "--dest_dir",
        type=str,
        required=True,
        help="Path to the destination directory.",
    )
    parser.add_argument(
        "-s",
        "--sample_rate",
        type=int,
        default=48000,
        help="Target sample rate of the audio files.",
    )
    parser.add_argument(
        "-min",
        "--min_duration",
        type=float,
        default=4.0,
        help="Lower bound of the duration of the audio files.",
    )
    parser.add_argument(
        "-max",
        "--max_duration",
        type=float,
        default=10.0,
        help="Duration of each segment.",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=40,
        help="Number of jobs to run in parallel.",
    )
    args = parser.parse_args()

    main(
        args.src_dir,
        args.dest_dir,
        args.sample_rate,
        args.min_duration,
        args.max_duration,
        args.num_workers,
    )
