import logging
import time
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import toml
import torch
import torch.backends.cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler  # type: ignore
from torch.cuda.amp import autocast  # type: ignore
from torch.nn.parallel import DistributedDataParallel
from torchinfo import summary
from tqdm import tqdm

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.logger import TensorboardLogger
from audiozen.utils import prepare_empty_dir

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer class."""

    def __init__(
        self,
        rank,
        config,
        resume,
        model,
        loss_function,
        optimizer,
        lr_scheduler,
    ) -> None:
        self.args = config

        # Setup directories
        self._initialize_exp_dirs_and_paths(config)

        # Control reproducibility
        if config["meta"]["use_deterministic_algorithms"]:
            self._use_deterministic_algorithms()

        # GPU
        self.rank = rank
        self.device = rank  # alias of rank

        # Distributed data parallel
        self.model = DistributedDataParallel(
            model.to(rank), device_ids=[rank], output_device=rank
        )

        # Optimizer
        self.optimizer = optimizer

        # LR scheduler
        self.lr_scheduler = lr_scheduler

        # Loss function
        self.loss_function = loss_function

        # Acoustic args
        self._setup_acoustic_args(config["acoustics"])

        # Automatic mixed precision (AMP)
        self.use_amp = config["meta"]["use_amp"]
        self.scaler = GradScaler(enabled=self.use_amp)

        # Trainer.train args
        self.train_config = config["trainer"]["train"]
        self.max_epoch = self.train_config["max_epoch"]
        self.clip_grad_norm_value = self.train_config["clip_grad_norm_value"]
        self.save_max_score = self.train_config["save_max_score"]
        self.save_checkpoint_interval = self.train_config["save_checkpoint_interval"]
        self.patience = self.train_config["patience"]

        # Trainer.validation args
        self.validate_config = config["trainer"]["validate"]
        self.validation_interval = self.validate_config["validation_interval"]
        self.max_num_checkpoints = self.validate_config["max_num_checkpoints"]
        assert (
            self.validation_interval >= 1
        ), "`validation_interval` should be large than one."

        # Other
        self.start_epoch = 1  # used when resuming
        self.current_epoch = 1  # used in custom training loop
        self.wait_count = 0
        self.best_score = -np.inf if self.save_max_score else np.inf
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        # Resume
        if resume:
            self._load_checkpoint("latest")

        if rank == 0:
            prepare_empty_dir(
                [
                    self.save_dir,
                    self.exp_dir,
                    self.checkpoints_dir,
                    self.tb_log_dir,
                    self.enhanced_dir,
                ],
                resume=resume,
            )

            self.writer = TensorboardLogger(self.tb_log_dir.as_posix())
            self.writer.log_config(config)

            with open(self.config_path.as_posix(), "w") as handle:
                toml.dump(config, handle)

            logger.info(
                f"Configuration file is saved to {self.config_path.as_posix()}."
            )

            # Backup of project code
            # shutil.copytree(src=self.source_code_dir.as_posix(), dst=self.source_code_backup_dir.as_posix())
            # logger.info(f"Project code is backed up to {self.source_code_backup_dir.as_posix()}.")

            # Model summary
            model_summary = summary(self.model, verbose=0)  # no output
            logger.info(f"\n {model_summary}")

    def _hf_search_setup(self, trial):
        """Setup hyperparameters for hyperparameter search.

        Args:
            trial: a trial object from ray.
        """
        for key, value in trial:
            if not hasattr(self.args, key):
                raise ValueError(f"Invalid hyperparameter {key} in the search space.")

            old_attr = getattr(self.args, key)

            # cast to the same type
            if old_attr is not None:
                value = type(old_attr)(value)

            setattr(self.args, key, value)

    @staticmethod
    def _get_time_now():
        return time.strftime("%Y_%m_%d--%H_%M_%S")

    def _initialize_exp_dirs_and_paths(self, config):
        """Initialize directories.

        Args:
            save_dir: the root directory to save all experiments.
            exp_id: the experiment id.

        Notes:
            - save_dir: /home/xhao/exp
            - exp_dir: /home/xhao/exp/fullsubnet_lr_0.1
            - checkpoints_dir: /home/xhao/exp/fullsubnet_lr_0.1/checkpoints
            - tb_log_dir: /home/xhao/exp/fullsubnet_lr_0.1/tb_log
            - src_source_code_dir: /home/xhao/audiozen
            - source_code_backup_dir: /home/xhao/exp/fullsubnet_lr_0.1/source_code__2023_01_07__17_19_57
            - config_path: /home/xhao/exp/fullsubnet_lr_0.1/config__2023_01_07__17_19_57.toml
        """
        self.save_dir = Path(config["meta"]["save_dir"]).expanduser().absolute()
        self.exp_dir = self.save_dir / config["meta"]["exp_id"]

        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.tb_log_dir = self.exp_dir / "tb_log"
        self.enhanced_dir = self.exp_dir / "enhanced"

        # Each run will have a unique source code, config, and log file.
        time_now = self._get_time_now()
        self.source_code_dir = (
            Path(__file__).expanduser().absolute().parent.parent.parent
        )
        self.source_code_backup_dir = self.exp_dir / f"source_code__{time_now}"
        self.config_path = self.exp_dir / f"config__{time_now}.toml"

    def _load_state_dict(self, checkpoint):
        if type(checkpoint) == dict:
            print("Loading checkpoint in dictionary...")
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_score = checkpoint["best_score"]
            self.wait_count = checkpoint["wait_count"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint["model"])
        else:
            print("Loading checkpoint in pth...")
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)

    def _load_checkpoint(self, ckpt_path="latest"):
        """load a checkpoint from the checkpints directory.

        Args:
            ckpt_path: "best", "latest", or a path to a checkpoint file
        """
        if ckpt_path == "best":
            ckpt_path = self.checkpoints_dir / "best.tar"
        elif ckpt_path == "latest":
            ckpt_path = self.checkpoints_dir / "latest.tar"
        else:
            ckpt_path = Path(ckpt_path).expanduser().absolute()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint {ckpt_path} does not exist.")

        # Load it on the CPU and later use .to(device) on the model
        # May slightly slow than using map_location="cuda:<...>"
        # https://stackoverflow.com/questions/61642619/pytorch-distributed-data-parallel-confusion
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        self._load_state_dict(checkpoint)

        if self.rank == 0:
            logger.info(f"Model checkpoint on epoch {self.start_epoch - 1} loaded.")

    def _set_attr_from_dict(self, config):
        """Set attributes from config."""
        for key, value in config.items():
            setattr(self, key, value)

    def _setup_acoustic_args(self, acoustic_args):
        """Setup acoustic arguments."""
        self.n_fft = acoustic_args["n_fft"]
        self.hop_length = acoustic_args["hop_length"]
        self.win_length = acoustic_args["win_length"]
        self.sr = acoustic_args["sr"]

        # Support for torch and librosa stft
        self.torch_stft = partial(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        self.torch_istft = partial(
            istft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        self.librosa_stft = partial(librosa.stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)  # fmt: skip
        self.librosa_istft = partial(librosa.istft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)  # fmt: skip

    @staticmethod
    def _use_deterministic_algorithms():
        """Control reproducibility of the training process.

        Notes:
            Deterministic operations are often slower than nondeterministic operations, see https://pytorch.org/docs/stable/notes/randomness.html.
        """
        # enable deterministic operations in CuDNN
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # enable deterministic operations in PyTorch
        torch.use_deterministic_algorithms(True)

    def _create_state_dict(self, epoch):
        """Create a state dict for saving."""
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "wait_count": self.wait_count,
            "model": self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict(),  # fmt: skip
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        return state_dict

    def _save_checkpoint(self, epoch, is_best_epoch):
        """Save checkpoint.

        Args:
            epoch: the current epoch.
            is_best_epoch: whether the current epoch is the best epoch.
        """
        state_dict = self._create_state_dict(epoch)
        checkpoint_fpath = self.checkpoints_dir / f"epoch_{str(epoch).zfill(4)}.tar"
        torch.save(state_dict, checkpoint_fpath.as_posix())
        torch.save(state_dict, self.checkpoints_dir / "latest.tar")

        if is_best_epoch:
            torch.save(state_dict, self.checkpoints_dir / "best.tar")

        # Find all regular checkpoints and only keep the latest `max_num_checkpoints` regular checkpoints
        checkpoint_paths = list(self.checkpoints_dir.glob("epoch_*.tar"))
        checkpoint_paths.sort()  # sort from oldest to newest

        if len(checkpoint_paths) > self.max_num_checkpoints:
            checkpoints_to_delete = checkpoint_paths[
                : len(checkpoint_paths) - self.max_num_checkpoints
            ]
            logger.info(f"Deleting {len(checkpoints_to_delete)} checkpoints...")
            for checkpoint in checkpoints_to_delete:
                checkpoint.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint.as_posix()}")

    def _build_step_kwargs(self, batch, batch_idx, dataloader_idx, dataloaders):
        step_kwargs = {
            "batch": batch,
            "batch_idx": batch_idx,
        }
        if len(dataloaders) > 1:
            step_kwargs["dataloader_idx"] = dataloader_idx

        return step_kwargs

    def _run_early_stop_check(self, score, epoch):
        should_stop = False

        if self._check_improvement(score, save_max_score=self.save_max_score):
            logger.info(f"Found new best score: {score:.4f}, saving checkpoint...")
            self._save_checkpoint(epoch, is_best_epoch=True)
            self.wait_count = 0
        else:
            logger.info(f"Score did not improve from {self.best_score:.4f}.")
            self.wait_count += 1
            logger.info(
                f"Early stopping counter: {self.wait_count} out of {self.patience}"
            )

            if self.wait_count >= self.patience:
                logger.info(f"Early stopping triggered, stopping training...")
                should_stop = True

        return should_stop

    def train(self, train_dataloader, validation_dataloaders):
        early_stop_mark = torch.zeros(1, device=self.rank)

        for epoch in range(self.start_epoch, self.max_epoch):
            self.current_epoch = epoch

            if self.rank == 0:
                logger.info(f"{'=' * 15} {epoch} epoch {'=' * 15}")
                logger.info("Begin training...")

            # Calling the set_epoch() method at the beginning of each epoch before
            # creating the DataLoader iterator is necessary to make shuffling work
            # properly across multiple epochs.
            train_dataloader.sampler.set_epoch(epoch)
            self.model.train()

            training_epoch_output = []

            if self.rank == 0:
                dataloader_bar = tqdm(
                    train_dataloader, desc="Train", dynamic_ncols=True
                )
            for batch_idx, batch in (
                enumerate(dataloader_bar)
                if self.rank == 0
                else enumerate(train_dataloader)
            ):
                # clear gradients
                self.optimizer.zero_grad()

                # forward with AMP
                with autocast(enabled=self.use_amp):
                    loss = self.training_step(batch, batch_idx)

                # backward
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)  # type: ignore
                self.scaler.step(self.optimizer)
                self.scaler.update()

                training_epoch_output.append(loss.item())

                if self.rank == 0:
                    dataloader_bar.set_description(
                        f"Loss: {loss.item():.4f}, lr: {self.lr_scheduler.get_last_lr()[-1]:.6f}"
                    )
                    self.writer.add_scalar(
                        f"Loss/Train_Step",
                        loss.item(),
                        epoch * len(train_dataloader) + batch_idx,
                    )

            self.training_epoch_end(training_epoch_output)

            if self.rank == 0:
                with torch.no_grad():
                    if epoch % self.save_checkpoint_interval == 0:
                        self._save_checkpoint(epoch, is_best_epoch=False)

                    if epoch % self.validation_interval == 0:
                        logger.info(f"Training finished, begin validation...")

                        score = self.validate(validation_dataloaders)

                        should_stop = self._run_early_stop_check(score, epoch)

                        if should_stop:
                            early_stop_mark += 1

                        logger.info(f"Validation finished.")

            dist.barrier()
            self.lr_scheduler.step()

            dist.all_reduce(early_stop_mark, op=dist.ReduceOp.SUM)

            if early_stop_mark != 0:
                # Call it out of DDP training loop to avoid hanging
                break

    @torch.no_grad()
    def validate(self, dataloaders):
        if self.rank == 0:
            logger.info(f"Begin validation...")
            self.model.eval()

            if not isinstance(dataloaders, list):
                dataloaders = [dataloaders]

            validation_output = []
            for dataloader_idx, dataloader in enumerate(dataloaders):
                dataloader_output = []
                for batch_idx, batch in enumerate(
                    tqdm(
                        dataloader,
                        desc=f"Inferring on dataloader {dataloader_idx}",
                        dynamic_ncols=True,
                    )
                ):
                    step_output = self.validation_step(batch, batch_idx, dataloader_idx)
                    dataloader_output.append(step_output)
                validation_output.append(dataloader_output)

            logger.info(f"Validation inference finished, begin validation epoch end...")
            score = self.validation_epoch_end(validation_output)

            return score

    @torch.no_grad()
    def test(self, dataloaders, ckpt_path="best"):
        """Test the model.

        Args:
            test_dataloaders: the dataloader(s) to test.
            ckpt_path: the checkpoint path to load the model weights from.
        """
        if self.rank == 0:
            logger.info(f"Begin testing...")
            self.model.eval()

            if not isinstance(dataloaders, list):
                dataloaders = [dataloaders]

            self._load_checkpoint(ckpt_path)

            test_output = []
            for dataloader_idx, dataloader in enumerate(dataloaders):
                step_outputs = []
                for batch_idx, batch in enumerate(
                    tqdm(
                        dataloader,
                        desc=f"Inference on dataloader {dataloader_idx}",
                        dynamic_ncols=True,
                    )
                ):
                    step_output = self.test_step(batch, batch_idx, dataloader_idx)
                    step_outputs.append(step_output)

                test_output.append(step_outputs)

            self.test_epoch_end(test_output)

    def _check_improvement(self, score, save_max_score=True):
        """Check if the current model got the best metric score"""
        if save_max_score:
            if score > self.best_score:
                self.best_score = score
                return True
            else:
                return False
        else:
            if score < self.best_score:
                self.best_score = score
                return True
            else:
                return False

    def training_step(self, batch, batch_idx):
        """Custom training step.

        Implement your own training step here. The input batch is from a training dataloader.
        The output of this function should be a loss tensor.

        Args:
            batch: a batch of data, which passed from a training dataloader.
            batch_idx: the index of the batch.

        Returns:
            loss: the loss of the batch.
        """
        raise NotImplementedError

    def training_epoch_end(self, training_epoch_output):
        """Training epoch end.

        When the training epoch ends, this function will be called.
        The input is a list of the loss value of each batch in the training epoch.
        You may want to log the epoch-level training loss here.

        Args:
            training_epoch_output: the output of the training epoch. It may a list of the output of each batch.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Validation step.

        This function defines the validation step. The input batch is from a validation dataloader.

        Args:
            batch: a batch of data.
            batch_idx: the index of the batch.
            dataloader_idx: the index of the dataloader.

        Returns:
            output: the output of the batch. It may enhanced audio signals.
        """
        raise NotImplementedError

    def validation_epoch_end(self, validation_epoch_output):
        """Validation epoch end.

        The input `validation_epoch_output` will be a list of list. For example, if you have two dataloaders, the validation_epoch_output will be

        ```python
        [
            [dataloader_1_batch_1_output, dataloader_1_batch_2_output, ...],
            [dataloader_2_batch_1_output, dataloader_2_batch_2_output, ...],
            ...
        ]
        ```

        The output of this function should be a metric score, which will be used to determine whether the current model is the best model.

        Args:
            validation_epoch_output: the output of the validation epoch. It is a list of list.

        Returns:
            score: the metric score of the validation epoch.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx, dataloader_idx):
        """Similar to validation_step, but for testing."""
        raise NotImplementedError

    def test_epoch_end(self, test_epoch_output):
        """Similar to validation_epoch_end, but for testing."""
        raise NotImplementedError
