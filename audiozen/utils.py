import importlib
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from audiozen.acoustics.audio_feature import loudness_rms_norm

logger = logging.getLogger(__name__)


def check_same_shape(est: Tensor, ref: Tensor) -> None:
    if est.shape != ref.shape:
        raise RuntimeError(f"Dimension mismatch: {est.shape=} vs {ref.shape=}.")


class Timer:
    """Count execution time.

    Examples:
        >>> timer = ExecutionTime()  # Start timer
        >>> print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.perf_counter()

    def duration(self, ndigits=3):
        """Get duration of execution.

        Args:
            ndigits: number of digits to round. Default: 3.
        """
        duration = round(time.perf_counter() - self.start_time, ndigits)
        return duration


def initialize_ddp(rank: int):
    """Initialize the process group"""
    torch.cuda.set_device(rank)

    # torchrun and multi-process distributed (single-node or multi-node) GPU training currently only achieves the best performance using the NCCL distributed backend.
    # The environment variables necessary to initialize a Torch process group are provided to you by this module, and no need for you to pass ``RANK`` manually.
    dist.init_process_group(backend="nccl")

    print(f"Initialized DistributedDataParallel process group on GPU {rank}.")


def instantiate(path: str, args: Optional[dict] = None, initialize: bool = True):
    """Load module or callable (like function) dynamically based on config string.

    Assume that the config items are as follows:

        [model]
            path = "model.FullSubNetModel"
            [model.args]
            n_frames = 32
            ...

    This function will:
        1. Load the "model" module from python search path.
        2. Load "model.FullSubNetModel" class or callable in the "model" module.
        3. If the "initialize" is set to True, instantiate (or call) class (or callable) with args (in "[model.args]").

    Args:
        path: Target class or callable path.
        args: Named arguments passed to class or callable.
        initialize: whether to initialize with args.

    Returns:
        If initialize is True, return the instantiated class or the return of the call.
        Otherwise, return the found class or callable

    Examples:
        >>> # Use official loss function
        >>> instantiate("torch.nn.CrossEntropyLoss", args={"label_smoothing": 0.2}, initialize=True)
        >>> # Use official optimizer
        >>> instantiate("torch.optim.Adam", args={"lr": 1e-3}, initialize=True)
        >>> # Use custom model in a recipe
        >>> instantiate("fsb.model.FullSubNetModel", args={"n_frames": 32}, initialize=True)
        >>> # Use custom loss function in audiozen
        >>> instantiate("audiozen.loss.CRMLoss", initialize=False)
    """
    # e.g., path = "fsb.model.FullSubNetModel"
    # module_path = "fsb.model"
    # class_or_function_name = "FullSubNetModel"
    splitted_path = path.split(".")

    if len(splitted_path) < 2:
        raise ValueError(f"Invalid path: {path}.")

    module_path = ".".join(splitted_path[:-1])
    class_or_function_name = splitted_path[-1]

    module = importlib.import_module(module_path)
    class_or_function = getattr(module, class_or_function_name)

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function


def set_random_seed(seed=3407):
    """Set random seed for reproducibility.

    Note:
        This function is used to control the reproducibility of the training process. Why the default value is 3407? It's just a joke.
        See https://arxiv.org/pdf/2109.08203.pdf.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def prepare_empty_dir(dirs, resume=False):
    """Prepare empty dirs.

    If resume a experiment, this function only assert that dirs should be exist.
    If does not resume a experiment, this function will set up new dirs.

    Args:
        dirs: a list of Path objects.
        resume: whether to resume a experiment. Default: False.
    """
    for dir_path in dirs:
        if resume:
            assert (
                dir_path.exists()
            ), "In resume mode, you must have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def activity_detector(
    audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=1e-6
):
    """Return the percentage of the time the audio signal is above an energy threshold
    Args:
        audio:
        fs:
        activity_threshold:
        target_level:
        eps:
    Returns:
    """
    audio, _, _ = loudness_rms_norm(audio, lvl=target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win**2) + eps)
        frame_energy_prob = 1.0 / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (
                1 - alpha_att
            )
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (
                1 - alpha_rel
            )

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active
