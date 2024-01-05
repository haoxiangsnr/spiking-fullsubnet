from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray


def load_audio(
    path: Union[Path, str], duration: Optional[float] = None, sr: Optional[int] = None, mode: str = "wrap", **kwargs
) -> Tuple[np.ndarray, int]:
    """Load the audio using soundfile and resample if necessary.

    Compared to sf.read, this function supports:
    - loading a segment of the audio when duration is specified.
    - resampling the audio when sr is specified.

    Args:
        path: Path to the audio file.
        duration: Duration of the audio segment in seconds. If None, the whole audio is loaded.
        sr: Sampling rate of the audio. If None, the original sampling rate is used.
        mode: Padding mode when duration is longer than the original audio.
        **kwargs: Additional keyword arguments for `np.pad`.

    Returns:
        A tuple of the audio signal and the sampling rate.

    Examples:
        >>> load_audio("test.wav", duration=2.0, sr=16000)
    """
    audio_path = str(path)

    with sf.SoundFile(path) as sf_desc:
        orig_sr = sf_desc.samplerate

        if duration is not None:
            frame_orig_duration = sf_desc.frames
            frame_duration = int(duration * orig_sr)

            if frame_duration < frame_orig_duration:
                # Randomly select a segment
                offset = np.random.randint(frame_orig_duration - frame_duration)
                sf_desc.seek(offset)
                y = sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=True).T
            else:
                y = sf_desc.read(dtype=np.float32, always_2d=True).T  # [C, T]
                if frame_duration > frame_orig_duration:
                    y = np.pad(y, ((0, 0), (0, frame_duration - frame_orig_duration)), mode=mode, **kwargs)
        else:
            y = sf_desc.read(dtype=np.float32, always_2d=True).T

    if y.shape[0] == 1:
        y = y.flatten()

    if sr is not None and sr != orig_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        orig_sr = sr

    return y, orig_sr


def subsample(
    data: np.ndarray, subsample_length: int, start_idx: int = -1, return_start_idx: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Sample a segment from the one-dimensional data.

    Args:
        data: One-dimensional data from which to sample.
        subsample_length: The length of the segment to be sampled.
        start_idx: The start index of the segment to be sampled. If less than 0, a random index is generated.
        return_start_idx: Whether to return the start index along with the data segment.

        Returns:
        The sampled data segment, and optionally the start index.

    Raises:
        ValueError: If data is not 1D, or if subsample_length is negative.

    Examples:
        >>> subsample(np.zeros(10), 5)
        >>> subsample(np.zeros(10), 5, start_idx=0)
        >>> subsample(np.zeros(10), 5, start_idx=0, return_start_idx=True)
    """
    if np.ndim(data) != 1:
        raise ValueError(f"Only support 1D data. The dim is {np.ndim(data)}")
    if subsample_length < 0:
        raise ValueError("subsample_length must be non-negative")

    data_len = len(data)

    if data_len > subsample_length:
        if start_idx < 0:
            start_idx = np.random.randint(data_len - subsample_length)
        end_idx = start_idx + subsample_length
        data = data[start_idx:end_idx]
    elif data_len < subsample_length:
        padding = subsample_length - data_len
        data = np.pad(data, (0, padding), "constant")
        start_idx = 0  # When padding, the start_idx is effectively 0

    if return_start_idx:
        return data, start_idx
    else:
        return data
