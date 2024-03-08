import logging
import sys
from dataclasses import dataclass
from typing import Union

from simple_parsing.helpers import Serializable


logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs(Serializable):
    """The arguments for the trainer."""

    output_dir: str
    do_train: bool = True  # Whether to run the training.
    do_eval: bool = False  # Whether to run the evaluation.
    do_predict: bool = False  # Whether to run the prediction.
    resume_from_checkpoint: Union[str, bool] = False  # The checkpoint to resume from.
    seed: int = 20220815  # The seed to use for reproducibility.

    per_device_train_batch_size: int = 8
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ddp_find_unused_parameters: bool = False

    dataloader_drop_last: bool = True
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = None

    debug: bool = False
    num_train_epochs: int = 3  # The maximum number of epochs to run.
    max_steps: int = 0  # The maximum number of steps to run.
    max_grad_norm: float = 1.0  # The maximum gradient norm to clip.
    greater_is_better: bool = True  # Whether the metric is better when greater.
    save_epoch_interval: int = 1  # The interval to save the checkpoint.
    eval_epoch_interval: int = 1  # The interval to validate the model.
    early_stopping_patience: int = 20  # The patience for early stopping.
    save_total_limit: int = sys.maxsize  # The maximum number of checkpoints to keep.

    lr_scheduler_type: str = "constant_schedule_with_warmup"  # The type of learning rate scheduler.
    warmup_steps: int = 0  # Linear warmup over warmup_steps.
    warmup_ratio: float = 0.0  # Linear warmup over warmup_ratio * max_steps.

    plot_lr: bool = False  # Whether to plot the learning rate.

    acoustic_n_fft: int = 512  # The number of FFT points.
    acoustic_hop_length: int = 256
    acoustic_win_length: int = 512
    acoustic_sr: int = 16000  # The sample rate of the audio.

    optim: str = "adamw"  # The optimizer to use.
    learning_rate: float = 1e-3  # The learning rate.
    adam_beta1: float = 0.9  # The beta1 for Adam.
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
