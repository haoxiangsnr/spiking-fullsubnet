import math
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import librosa
import pandas as pd
import toml
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.debug_utils import DebugUnderflowOverflow
from audiozen.logger import TensorboardLogger
from audiozen.optimization import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from audiozen.trainer_utils import TrainerState
from audiozen.utils import prepare_empty_dir, print_env

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        accelerator: Accelerator,
        config,
        resume,
        model,
        optimizer,
        loss_function,
    ):
        """Create an instance of BaseTrainer for training, validation, and testing."""
        # Setup directories
        self._initialize_exp_dirs_and_paths(config)

        # GPU
        self.accelerator = accelerator
        self.rank = accelerator.device
        self.device = accelerator.device  # alias of rank

        # Model
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        # Acoustic args
        self._setup_acoustic_args(config["acoustics"])

        # Trainer.train args
        self.trainer_config = config["trainer"]["args"]
        self.debug = self.trainer_config.get("debug", False)
        self.max_steps = self.trainer_config.get("max_steps", 0)
        self.max_epochs = self.trainer_config.get("max_epochs", sys.maxsize)
        self.max_grad_norm = self.trainer_config.get("max_grad_norm", 0)
        self.save_max_score = self.trainer_config.get("save_max_score", True)
        self.save_ckpt_interval = self.trainer_config.get("save_ckpt_interval", 1)
        self.max_patience = self.trainer_config.get("max_patience", 10)
        self.plot_norm = self.trainer_config.get("plot_norm", True)
        self.validation_interval = self.trainer_config.get("validation_interval", 1)
        self.max_num_checkpoints = self.trainer_config.get("max_num_checkpoints", 10)
        self.scheduler_name = self.trainer_config.get("scheduler_name", "constant_schedule_with_warmup")
        self.warmup_steps = self.trainer_config.get("warmup_steps", 0)
        self.warmup_ratio = self.trainer_config.get("warmup_ratio", 0.0)
        self.gradient_accumulation_steps = self.trainer_config.get("gradient_accumulation_steps", 1)

        if self.max_steps > 0:
            logger.info(f"`max_steps` is set to {self.max_steps}. Ignoring `max_epochs`.")

        if self.validation_interval < 1:
            logger.info(f"`validation_interval` is set to {self.validation_interval}. It must be >= 1.")

        # Trainer states
        self.state = TrainerState(save_max_score=self.save_max_score)
        self.accelerator.register_for_checkpointing(self.state)  # Register accelerate objects

        # Others
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        # Resume
        if resume:
            self._load_checkpoint(ckpt_path="latest")

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

        logger.info(f"Environment information:\n{print_env()}")

        # Backup of project code
        # shutil.copytree(src=self.source_code_dir.as_posix(), dst=self.source_code_backup_dir.as_posix())
        # logger.info(f"Project code is backed up to {self.source_code_backup_dir.as_posix()}.")

        # Model summary
        logger.info(f"\n {summary(self.model, verbose=0)}")

    def _run_early_stop_check(self, score: float):
        should_stop = False

        if self._check_improvement(score, save_max_score=self.save_max_score):
            self.state.best_score = score
            self.state.best_score_epoch = self.state.epochs_trained
            self._save_checkpoint(self.state.epochs_trained, is_best_epoch=True)
            self.state.patience = 0
            logger.info(f"Found new best score: {score:.4f}, saving checkpoint...")
        else:
            logger.info(
                f"Score did not improve from {self.state.best_score:.4f} at epoch {self.state.best_score_epoch}."
            )
            self.state.patience += 1
            logger.info(f"Early stopping counter: {self.state.patience} out of {self.max_patience}")

            if self.state.patience >= self.max_patience:
                logger.info(f"Early stopping triggered, stopping training...")
                should_stop = True

        return should_stop

    def _setup_acoustic_args(self, acoustic_args):
        """Setup acoustic arguments."""
        n_fft = acoustic_args["n_fft"]
        hop_length = acoustic_args["hop_length"]
        win_length = acoustic_args["win_length"]
        sr = acoustic_args["sr"]

        # Support for torch and librosa stft
        self.torch_stft = partial(stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.torch_istft = partial(istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.librosa_stft = partial(librosa.stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.librosa_istft = partial(librosa.istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sr = sr

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

    def _find_latest_ckpt_path(self):
        """Find the latest checkpoint path."""
        # Pick up all checkpoints with the format `epoch_*`
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_" + ("[0-9]" * 4)))

        # Remove files that is not a checkpoint
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.is_dir()]

        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoints_dir.as_posix()}.")

        # Pick up the latest checkpoint
        ckpt_path = checkpoints[-1]

        return ckpt_path

    def _load_checkpoint(self, ckpt_path):
        """load a checkpoint from the checkpints directory.

        Args:
            ckpt_path: "best", "latest", or a path to a checkpoint file
        """
        if ckpt_path == "best":
            ckpt_path = self.checkpoints_dir / "best"
        elif ckpt_path == "latest":
            ckpt_path = self._find_latest_ckpt_path()
        else:
            ckpt_path = Path(ckpt_path).expanduser().absolute()

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path.as_posix()} not found.")

        self.accelerator.load_state(ckpt_path, map_location="cpu")

        logger.info(f"Checkpoint on epoch {self.state.epochs_trained} is loaded.")

    def _save_checkpoint(self, epoch, is_best_epoch):
        """Save checkpoint.

        Args:
            epoch: the current epoch.
            is_best_epoch: whether the current epoch is the best epoch.
        """
        # Save checkpoint
        if is_best_epoch:
            self.accelerator.save_state(self.checkpoints_dir / "best")
        else:
            # Regular checkpoint
            ckpt_path = self.checkpoints_dir / f"epoch_{str(epoch).zfill(4)}"
            self.accelerator.save_state(ckpt_path.as_posix())

        # Find all regular checkpoints and only keep the latest `max_num_checkpoints` regular checkpoints
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_*"))

        if epoch <= len(checkpoints):
            logger.warning(
                f"Current epoch is {epoch}, but found {len(checkpoints)} checkpoints. "
                f"This may be caused by you running the same experiment multiple times. "
                f"Recommend to run the experiment with a different `exp_id`."
            )

        if len(checkpoints) > self.max_num_checkpoints:
            logger.info(
                f"Found {len(checkpoints)} checkpoints, only keeping the latest {self.max_num_checkpoints} checkpoints."
            )
            for checkpoint_dir in checkpoints[: -self.max_num_checkpoints]:
                shutil.rmtree(checkpoint_dir.as_posix())
                logger.info(f"Checkpoint {checkpoint_dir.as_posix()} is removed.")

    @staticmethod
    def get_warmup_steps(warmup_steps, max_steps, warmup_ratio):
        if warmup_steps > 0:
            logger.info(f"warmup_steps={warmup_steps}. warmup_ratio will be ignored.")
            return warmup_steps
        else:
            return math.ceil(max_steps * warmup_ratio)

    def create_warmup_scheduler(self, optimizer, scheduler_name, max_steps: int):
        num_warmup_steps = self.get_warmup_steps(self.warmup_steps, max_steps, self.warmup_ratio)
        if scheduler_name == "constant_schedule_with_warmup":
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif scheduler_name == "linear_schedule_with_warmup":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_steps
            )

    def create_schedulers(self, max_steps: int):
        """Create schedulers.

        You can override this method to create your own schedulers. For example, in GAN training, you may want to
        create two schedulers for the generator and the discriminator.

        Args:
            max_steps: the maximum number of steps to train.
        """
        self.lr_scheduler = self.create_warmup_scheduler(
            optimizer=self.optimizer, scheduler_name=self.scheduler_name, max_steps=max_steps
        )
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

    def set_models_to_train_mode(self):
        """Set models to train mode.

        You can override this method to set your own models to train mode. For example, in GAN training, you may want to
        set the generator and the discriminator to train mode.
        """
        self.model.train()

    def set_models_to_eval_mode(self):
        self.model.eval()

    def lr_scheduler_step(self):
        """Step the lr scheduler.

        You can override this method to step your own lr scheduler. For example, in GAN training, you may want to
        step the lr scheduler of the generator and the discriminator.
        """
        self.lr_scheduler.step(self.state.steps_trained)

    def create_bar_desc(self, loss_dict, norm):
        bar_desc = ""
        for k, v in loss_dict.items():
            bar_desc += f"{k}: {(v):.4f}, "
        bar_desc += f"norm: {norm:.4f}, " f"lr: {self.lr_scheduler.get_last_lr()[-1]:.10f}"
        return bar_desc

    def train(self, train_dataloader: DataLoader, validation_dataloaders):
        """Train the model.

        Args:
            train_dataloader: the dataloader to train.
            validation_dataloaders: the dataloader(s) to validate.

        Notes:
            You are responsible for calling `.backward()`, `.step()`, and `.zero_grad()` in your implementation
            of `training_step()`. Accelerate will automatically handle the gradient accumulation for you.
            It means that in gradient accumulation, the step() of optimizer and scheduler is called only when gradient_accumulation_steps is reached.

            The training step is implemented as follows:

            .. code-block:: python

                    self.optimizer.zero_grad()
                    loss = training_step(batch, batch_idx)
                    self.accelerator.backward(loss)
                    self.optimizer.step()

                    return {
                        "loss": loss,
                    }
        """
        early_stop_mark = torch.zeros(1, device=self.device)

        if self.debug:
            logger.info("Debug mode is on")
            DebugUnderflowOverflow(self.model)

        # Setting up training control variables
        steps_per_epoch = len(train_dataloader)
        update_steps_per_epoch = steps_per_epoch // self.gradient_accumulation_steps
        update_steps_per_epoch = max(update_steps_per_epoch, 1)

        if self.max_steps > 0:
            max_steps = self.max_steps
            max_epochs = self.max_steps // update_steps_per_epoch + int(self.max_steps % update_steps_per_epoch > 0)
        else:
            max_steps = self.max_epochs * update_steps_per_epoch
            max_epochs = self.max_epochs

        logger.info("Training control variables:")
        logger.info(f"`steps_per_epoch`: {steps_per_epoch}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"`update_steps_per_epoch`: {update_steps_per_epoch}")
        logger.info(f"`max_steps`: {max_steps}")
        logger.info(f"`max_epochs`: {max_epochs}")

        # Generator learning rate scheduler
        self.create_schedulers(max_steps=max_steps)

        for epoch in range(self.state.epochs_trained + 1, max_epochs + 1):
            logger.info(f"{'=' * 9} Epoch {epoch} out of {max_epochs} {'=' * 9}")
            logger.info("Begin training...")

            self.set_models_to_train_mode()

            training_epoch_output = []

            # the iter number of progress bar increments by 1 by default whether gradient accumulation is used or not.
            # but we update the description of the progress bar only when the gradients are synchronized across all processes.
            dataloader_bar = tqdm(
                train_dataloader,
                desc="",
                dynamic_ncols=True,
                bar_format="{l_bar}{r_bar}",
                colour="green",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch_idx, batch in enumerate(dataloader_bar):
                # accumulate() will automatically skip synchronization if applicable loss is linearly scaled with the optimizer.grad
                # accumulate() will automatically divide the loss in backward by the number of gradient accumulation steps
                # However, it won't return this loss, so we need to manually divide the loss by the number of gradient accumulation steps.
                with self.accelerator.accumulate(self.model):
                    # You are responsible for calling `.backward()`, `.step()`, and `.zero_grad()` in your implementation
                    loss_dict = self.training_step(batch, batch_idx)

                    # I guess we don't need to divide the loss by the number of gradient accumulation steps here
                    # for visualization, we just plot the mean of mean of the loss of each batch
                    training_epoch_output.append(loss_dict)

                    # If `sync_gradients` is True, the gradients are currently being synced across all processes.
                    # It means that the current step we have finished the cumulative gradient accumulation.
                    if self.accelerator.sync_gradients:
                        # The gradients are added across all processes in this cumulative gradient accumulation step.
                        if self.max_grad_norm > 0:
                            norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        if self.accelerator.is_local_main_process:
                            bar_desc = self.create_bar_desc(loss_dict, norm)
                            dataloader_bar.set_description_str(bar_desc)

                    if not self.accelerator.optimizer_step_was_skipped:
                        # We can put the scheduler.step() into the training_step() function. However, it has **too much
                        # details should be considered**. It's better to put it here and add some comments.
                        #
                        # 1. every process lr_scheduler step N times, where N is the number of processes. We need to multiply the number of steps by the number of processes before constructing the
                        # scheduler to make sure it behaves as we expect it to do. https://github.com/huggingface/accelerate/issues/1398
                        #
                        # 2. For AMP, if the gradients are `nan` or `inf` skip the update step, we should call the
                        # `scheduler.step()` after checking `self.accelerator.optimizer_step_was_skipped`.
                        # Otherwise, the scheduler.step() will be called even if the optimizer step is skipped.
                        self.lr_scheduler_step()

                self.state.steps_trained += 1
            self.state.epochs_trained += 1
            self.training_epoch_end(training_epoch_output)

            # Should save, evaluate, and early stop?
            if self.accelerator.is_local_main_process and epoch % self.save_ckpt_interval == 0:
                self._save_checkpoint(epoch, is_best_epoch=False)

            if epoch % self.validation_interval == 0:
                with torch.no_grad():
                    logger.info(f"Training finished, begin validation...")
                    score = self.validate(validation_dataloaders)

                    if self.accelerator.is_local_main_process:
                        should_stop = self._run_early_stop_check(score)
                        if should_stop:
                            early_stop_mark += 1

                    logger.info(f"Validation finished.")

            self.accelerator.wait_for_everyone()

            # Reduces the `early_stop_mark` data across all processes
            # If `early_stop_mark` is 1 in any process, then `reduce_early_stop_mark` will be 1 in all processes.
            reduced_early_stop_mark = self.accelerator.reduce(early_stop_mark, reduction="sum")

            # If any process triggers early stopping, stop training
            if reduced_early_stop_mark != 0:
                break

    @torch.no_grad()
    def validate(self, dataloaders):
        logger.info(f"Begin validation...")

        self.set_models_to_eval_mode()

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
        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        self._load_checkpoint(ckpt_path)

        self.set_models_to_eval_mode()

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

            self.set_models_to_eval_mode()

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
            if score > self.state.best_score:
                return True
            else:
                return False
        else:
            if score < self.state.best_score:
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
