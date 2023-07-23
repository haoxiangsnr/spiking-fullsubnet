"""
Filter out audio files with high DNS MOS score.

Usage:
    python dns_mos_filter.py \
        --src_dir <src_dir> \
        --dest_dir <dest_dir> \
        --mos_threshold <mos_threshold> \

Note:
    "*.onnx" is required to run this script.
"""
from pathlib import Path

import click
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm

SAMPLERATE = 16000
INPUT_LENGTH = 9.01


def audio_melspec(
    audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
    )
    if to_db:
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


def dns_mos_p835(onnx_sess, audio):
    len_samples = int(INPUT_LENGTH * SAMPLERATE)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)

    num_hops = int(np.floor(len(audio) / SAMPLERATE) - INPUT_LENGTH) + 1
    hop_len_samples = SAMPLERATE
    predicted_mos_ovr_seg_raw = []

    for idx in range(num_hops):
        audio_seg = audio[
            int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
        ]
        if len(audio_seg) < len_samples:
            continue

        input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
        oi = {"input_1": input_features}
        *_, mos_ovr_raw = onnx_sess.run(None, oi)[0][0]
        predicted_mos_ovr_seg_raw.append(mos_ovr_raw)

    return np.mean(predicted_mos_ovr_seg_raw)


@click.command()
@click.argument("src_dir", type=click.Path(exists=True))
@click.argument("dest_dir", type=click.Path())
@click.option("--mos_threshold", type=float, default=4.25)
@click.option("--len_upper_threshold", type=float, default=60)
@click.option("--len_lower_threshold", type=float, default=3)
def main(
    src_dir,
    dest_dir,
    mos_threshold=4.25,
    len_upper_threshold=20,
    len_lower_threshold=3,
):
    print(
        f"Filtering audio files with MOS score higher than {mos_threshold}. length between {len_lower_threshold} and {len_upper_threshold}"
    )
    src_dir = Path(src_dir).expanduser().absolute()
    dest_dir = Path(dest_dir).expanduser().absolute()

    fpath_list = librosa.util.find_files(src_dir.as_posix(), ext=["wav"])
    orig_len = len(fpath_list)
    print(f"Found {orig_len} files in {src_dir.as_posix()}.")
    out_len = 0

    onnx_sess = ort.InferenceSession(
        "/home/xianghao/proj/audiozen/audiozen/external/DNSMOS/sig_bak_ovr.onnx",
        providers=["CUDAExecutionProvider"],
    )

    progress_bar = tqdm(fpath_list)
    for fpath in progress_bar:
        progress_bar.set_description(f"Filtered {out_len} files from {orig_len} files.")
        fpath = Path(fpath)
        dest_fpath = dest_dir / fpath.relative_to(src_dir)
        dest_fpath.parent.mkdir(parents=True, exist_ok=True)

        audio, _ = librosa.load(fpath, sr=SAMPLERATE)
        audio_len = len(audio) / SAMPLERATE

        if audio_len < len_lower_threshold or audio_len > len_upper_threshold:
            continue

        mos = dns_mos_p835(onnx_sess, audio)

        if mos < mos_threshold:
            continue

        sf.write(dest_fpath.as_posix(), audio, SAMPLERATE)
        out_len += 1

    print(f"Filtered {orig_len - out_len} files out of {orig_len} files.")


if __name__ == "__main__":
    main()
