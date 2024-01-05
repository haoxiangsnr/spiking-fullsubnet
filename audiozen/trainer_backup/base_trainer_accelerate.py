import shutil
import time
from functools import partial
from pathlib import Path

import librosa
import pandas as pd
import toml
import torch
import torch.backends.cudnn
from accelerate import Accelerator
from accelerate.logging import get_logger
from torchinfo import summary
from tqdm.auto import tqdm

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.logger import TensorboardLogger
from audiozen.trainer_backup.utils import BestScoreState, EpochState, WaitCountState
from audiozen.utils import prepare_empty_dir

logger = get_logger(__name__)


class BaseTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
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

        # GPU
        self.accelerator = accelerator
        self.rank = accelerator.device
        self.device = accelerator.device  # alias of rank

        # Model
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function

        # Acoustic args
        self._setup_acoustic_args(config["acoustics"])

        # Trainer.train args
        self.trainer_config = config["trainer"]["args"]
        self.max_epoch = self.trainer_config.get("max_epoch", 9999)
        self.clip_grad_norm_value = self.trainer_config.get("clip_grad_norm_value", -1)
        self.save_max_score = self.trainer_config.get("save_max_score", True)
        self.save_ckpt_interval = self.trainer_config.get("save_ckpt_interval", 1)
        self.patience = self.trainer_config.get("patience", 10)
        self.plot_norm = self.trainer_config.get("plot_norm", True)
        self.validation_interval = self.trainer_config.get("validation_interval", 1)
        self.max_num_checkpoints = self.trainer_config.get("max_num_checkpoints", 10)
        assert self.validation_interval >= 1, "'validation_interval' should be large than one."

        # Count Variables
        self.total_norm = -1
        self.start_epoch = EpochState()
        self.current_epoch = 1  # used in custom training loop
        self.wait_count = WaitCountState()
        self.best_score = BestScoreState(save_max_score=self.save_max_score)

        # Register accelerate objects
        self.accelerator.register_for_checkpointing(self.start_epoch)
        self.accelerator.register_for_checkpointing(self.wait_count)
        self.accelerator.register_for_checkpointing(self.best_score)

        # Others
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        # Resume
        if resume:
            self._load_checkpoint("latest")

        if self.accelerator.is_local_main_process:
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

        logger.info(f"Configuration file is saved to {self.config_path.as_posix()}.")

        # Backup of project code
        # shutil.copytree(src=self.source_code_dir.as_posix(), dst=self.source_code_backup_dir.as_posix())
        # logger.info(f"Project code is backed up to {self.source_code_backup_dir.as_posix()}.")

        # Model summary
        model_summary = summary(self.model, verbose=0)  # no output
        logger.info(f"\n {model_summary}")

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
        self.source_code_dir = Path(__file__).expanduser().absolute().parent.parent.parent
        self.source_code_backup_dir = self.exp_dir / f"source_code__{time_now}"
        self.config_path = self.exp_dir / f"config__{time_now}.toml"

    def _load_checkpoint(self, ckpt_path):
        """load a checkpoint from the checkpints directory.

        Args:
            ckpt_path: "best", "latest", or a path to a checkpoint file
        """
        if ckpt_path == "best":
            ckpt_path = self.checkpoints_dir / "best"
        elif ckpt_path == "latest":
            ckpt_path = self.checkpoints_dir / "latest"
        else:
            ckpt_path = Path(ckpt_path).expanduser().absolute()

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path.as_posix()} not found.")

        self.accelerator.load_state(ckpt_path)

        logger.info(f"Checkpoint on epoch {self.start_epoch.value} is loaded.")

    def _save_checkpoint(self, epoch, is_best_epoch):
        """Save checkpoint.

        Args:
            epoch: the current epoch.
            is_best_epoch: whether the current epoch is the best epoch.
        """
        self.start_epoch.value = epoch

        if is_best_epoch:
            self.accelerator.save_state(self.checkpoints_dir / "best")
        else:
            # Regular checkpoint
            ckpt_path = self.checkpoints_dir / f"epoch_{str(epoch).zfill(4)}"
            self.accelerator.save_state(ckpt_path.as_posix())
            self.accelerator.save_state(self.checkpoints_dir / "latest")

        # Find all regular checkpoints and only keep the latest `max_num_checkpoints` regular checkpoints
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_*"))
        if len(checkpoints) > self.max_num_checkpoints:
            logger.info(
                f"Found {len(checkpoints)} checkpoints, only keeping the latest {self.max_num_checkpoints} checkpoints."
            )
            for checkpoint_dir in checkpoints[: -self.max_num_checkpoints]:
                shutil.rmtree(checkpoint_dir.as_posix())
                logger.info(f"Checkpoint {checkpoint_dir.as_posix()} is removed.")

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
        self.librosa_stft = partial(
            librosa.stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        self.librosa_istft = partial(
            librosa.istft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _run_early_stop_check(self, score: float, epoch: int):
        should_stop = False

        if self._check_improvement(score, save_max_score=self.save_max_score):
            logger.info(f"Found new best score: {score:.4f}, saving checkpoint...")
            self._save_checkpoint(epoch, is_best_epoch=True)
            self.wait_count.value = 0
        else:
            logger.info(f"Score did not improve from {self.best_score.value:.4f}.")
            self.wait_count.value += 1
            logger.info(f"Early stopping counter: {self.wait_count.value} out of {self.patience}")

            if self.wait_count.value >= self.patience:
                logger.info(f"Early stopping triggered, stopping training...")
                should_stop = True

        return should_stop

    def train(self, train_dataloader, validation_dataloaders):
        early_stop_mark = torch.zeros(1, device=self.accelerator.device)

        for epoch in range(self.start_epoch.value, self.max_epoch + 1):
            self.current_epoch = epoch

            logger.info(f"{'=' * 15} Epoch {epoch} {'=' * 15}")
            logger.info("Begin training...")

            self.model.train()

            training_epoch_output = []

            dataloader_bar = tqdm(
                train_dataloader,
                desc="",
                dynamic_ncols=True,
                bar_format="{l_bar}{r_bar}",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch_idx, batch in enumerate(dataloader_bar):
                loss_dict = self.training_step(batch, batch_idx)
                training_epoch_output.append(loss_dict)

                if self.accelerator.is_local_main_process:
                    bar_desc = ""
                    for k, v in loss_dict.items():
                        bar_desc += f"{k}: {v:.4f}, "
                    bar_desc += f"lr: {self.lr_scheduler.get_last_lr()[-1]:.6f}, "
                    dataloader_bar.set_description(bar_desc)

                    # Log to tensorboard
                    # for key, value in loss_dict.items():
                    #     self.writer.add_scalar(
                    #         f"Train_Step/{key}",
                    #         value,
                    #         (epoch - 1) * len(train_dataloader) + batch_idx,
                    #     )

            self.training_epoch_end(training_epoch_output)

            if self.accelerator.is_local_main_process and epoch % self.save_ckpt_interval == 0:
                self._save_checkpoint(epoch, is_best_epoch=False)

            if epoch % self.validation_interval == 0:
                with torch.no_grad():
                    logger.info(f"Training finished, begin validation...")
                    score = self.validate(validation_dataloaders)

                    if self.accelerator.is_local_main_process:
                        should_stop = self._run_early_stop_check(score, epoch)
                        if should_stop:
                            early_stop_mark += 1

                    logger.info(f"Validation finished.")

            if not self.accelerator.optimizer_step_was_skipped:
                # For mixed precision training,
                # `optimizer_step_was_skipped` will be True if the gradients are `nan` or `inf`.
                self.lr_scheduler.step()
            self.accelerator.wait_for_everyone()

            # Reduces the `early_stop_mark` data across all processes in such a way that all get the final result.
            reduced_early_stop_mark = self.accelerator.reduce(early_stop_mark)

            # If any process triggers early stopping, stop training
            if reduced_early_stop_mark != 0:
                break

    @torch.no_grad()
    def validate(self, dataloaders):
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
                    desc=f"Inference on dataloader {dataloader_idx}",
                    bar_format="{l_bar}{r_bar}",
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_local_main_process,
                )
            ):
                step_output = self.validation_step(batch, batch_idx, dataloader_idx)

                # Note that the first dimension of the result is num_processes multiplied
                # by the first dimension of the input tensors.
                # =================================================
                # [noisy,       clean_y, ...]
                # [[4, 480000], [4, 480000], ...]
                # =================================================
                gathered_step_output = self.accelerator.gather_for_metrics(step_output)
                dataloader_output.append(gathered_step_output)

            validation_output.append(dataloader_output)

        logger.info(f"Validation inference finished, begin validation epoch end...")

        if self.accelerator.is_local_main_process:
            # only the main process will run validation_epoch_end
            score = self.validation_epoch_end(validation_output)
            return score
        else:
            return None

    @torch.no_grad()
    def test(self, dataloaders, ckpt_path="best"):
        """Test the model.

        Args:
            test_dataloaders: the dataloader(s) to test.
            ckpt_path: the checkpoint path to load the model weights from.
        """
        logger.info(f"Begin testing...")
        self.model.eval()

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        self._load_checkpoint(ckpt_path)

        test_output = []
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dataloader_out = []
            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Inference on dataloader {dataloader_idx}",
                    bar_format="{l_bar}{r_bar}",
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_local_main_process,
                )
            ):
                step_output = self.test_step(batch, batch_idx, dataloader_idx)

                # Note that the first dimension of the result is num_processes multiplied
                # by the first dimension of the input tensors.
                # =================================================
                # [noisy,       clean_y, ...]
                # [[4, 480000], [4, 480000], ...]
                # =================================================
                gathered_step_output = self.accelerator.gather_for_metrics(step_output)
                dataloader_out.append(gathered_step_output)

            test_output.append(dataloader_out)

        logger.info(f"Testing inference finished, begin testing epoch end...")
        if self.accelerator.is_local_main_process:
            # only the main process will run test_epoch_end
            self.test_epoch_end(test_output)

    @torch.no_grad()
    def predict(self, dataloaders, ckpt_path="best"):
        """Predict.

        Notes:
            In predict, there are no labels available, only the model inputs, meaning prediction isn't used for evaluation.

        Args:
            dataloaders: the dataloader(s) to predict.
            ckpt_path: the checkpoint path to load the model weights from.
        """
        if self.rank == 0:
            logger.info(f"Begin predicting...")
            self.model.eval()

            if not isinstance(dataloaders, list):
                dataloaders = [dataloaders]

            self._load_checkpoint(ckpt_path)

            for dataloader_idx, dataloader in enumerate(dataloaders):
                for batch_idx, batch in enumerate(
                    tqdm(
                        dataloader,
                        desc=f"Inference on dataloader {dataloader_idx}",
                        bar_format="{l_bar}{r_bar}",
                    )
                ):
                    self.predict_step(batch, batch_idx, dataloader_idx)

    def _check_improvement(self, score, save_max_score=True):
        """Check if the current model got the best metric score"""
        if save_max_score:
            if score > self.best_score.value:
                self.best_score.value = score
                return True
            else:
                return False
        else:
            if score < self.best_score.value:
                self.best_score.value = score
                return True
            else:
                return False

    def training_step(self, batch, batch_idx):
        """Implement a training step.

        Implement your own training step here.
        The input batch is from a training dataloader and the output of this function should be a loss tensor.
        Here is the persuade code for training a model:

        .. code-block:: python
            :emphasize-lines: 7

            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    zero_grad()
                    loss = training_step(batch, batch_idx)
                    loss.backward()
                    optimizer.step()

                training_epoch_output.append(loss)
                training_epoch_end(training_epoch_output)

                save_checkpoint()

                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)


        Args:
            batch: a batch of data, which passed from a custom training dataloader.
            batch_idx: the index of the current batch.

        Returns:
            loss: the loss of the batch.
        """
        raise NotImplementedError

    def training_epoch_end(self, training_epoch_output):
        """Implement the logic of the end of a training epoch.

        When the training epoch ends, this function will be called.
        The input is a list of the loss value of each batch in the training epoch.
        You may want to log the epoch-level training loss here.

        .. code-block:: python
            :emphasize-lines: 12

            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    zero_grad()
                    loss = training_step(batch, batch_idx)
                    loss.backward()
                    optimizer.step()

                training_epoch_output.append(loss)
                training_epoch_end(training_epoch_output)

                save_checkpoint()

                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)

        Args:
            training_epoch_output: the output of the training epoch. It may a list of the output of each batch.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Implement a validation step.

        This function defines the validation step. The input batch is from a validation dataloader.
        Here is the persuade code for validating a model:

        .. code-block:: python
            :emphasize-lines: 4

            validation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = validation_step(batch, batch_idx)
                    validation_epoch_output.append(loss_or_data)

            score = validation_epoch_end(validation_epoch_output)
            return score

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

        The input `validation_epoch_output` will be a list of list. For example, if you have two dataloaders, the `validation_epoch_output` will be:

        .. code-block:: python

            validation_epoch_output = [
                [dataloader_1_batch_1_output, dataloader_1_batch_2_output, ...],
                [dataloader_2_batch_1_output, dataloader_2_batch_2_output, ...],
                ...
            ]


        The output of this function should be a metric score, which will be used to determine whether the current model is the best model.

        .. code-block:: python
            :emphasize-lines: 7

            validation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = validation_step(batch, batch_idx)
                    validation_epoch_output.append(loss_or_data)

            score = validation_epoch_end(validation_epoch_output)
            return score

        Args:
            validation_epoch_output: the output of the validation epoch. It is a list of list.

        Returns:
            score: the metric score of the validation epoch.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx, dataloader_idx):
        """Similar to validation_step, but for testing.

        .. code-block:: python
            :linenos:
            :emphasize-lines: 4

            load_checkpoint(ckpt_path)

            for batch, batch_index in dataloader:
                loss = test_step(batch, batch_idx)

                test_epoch_output.append(loss)

            test_epoch_end(test_epoch_output)

            return score
        """
        raise NotImplementedError

    def test_epoch_end(self, test_epoch_output):
        """Similar to validation_epoch_end, but for testing.

        .. code-block:: python
            :linenos:
            :emphasize-lines: 8

            load_checkpoint(ckpt_path)

            for batch, batch_index in dataloader:
                loss = test_step(batch, batch_idx)

                test_epoch_output.append(loss)

            test_epoch_end(test_epoch_output)

            return score
        """
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx):
        """Similar to validation_step, but for predict.

        .. code-block:: python
            :linenos:
            :emphasize-lines: 4

            load_checkpoint(ckpt_path)

            for batch, batch_index in dataloader:
                loss = predict_step(batch, batch_idx)

                predict_epoch_output.append(loss)
        """
        pass
