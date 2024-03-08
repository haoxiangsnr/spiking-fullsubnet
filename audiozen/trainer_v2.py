import math
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import librosa
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, set_seed
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm.auto import tqdm

from audiozen.acoustics.audio_feature import istft, stft
from audiozen.debug_utils import DebugUnderflowOverflow
from audiozen.logger import TensorboardLogger
from audiozen.optimization import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from audiozen.trainer_args import TrainingArgs
from audiozen.trainer_utils import TrainerState, seed_worker
from audiozen.utils import prepare_empty_dir, print_env


logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArgs,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        """Initialize the trainer.

        Args:
            model: The model to train, evaluate or use for predictions.
            args: The training arguments.
            data_collator: The data collator to use for training and evaluation. Defaults to None.
            train_dataset: The training dataset. Defaults to None.
            eval_dataset: The evaluation dataset. Defaults to None. It can be a single dataset or a dictionary of datasets.
            optimizers: The optimizer and the learning rate scheduler. Defaults to (None, None).

        Important attributes:
            **is_in_train**: Whether or not a model is currently running `train` (e.g. when `evaluate` is called while in `train`)
        """
        self.args = args
        self.is_in_train = False
        # Accelerator should be created as early as possible
        self.create_accelerator_and_postprocess()

        # Set random seed using the method from accelerate
        set_seed(self.args.seed, device_specific=True)

        # Model. We are able to prepare the model directly in the __init__ method. However, we don't do it there is a bug
        self.model = model

        # Optimizers
        self.optimizer, self.lr_scheduler = optimizers

        # Datasets
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Setup directories
        self._setup_exp_paths(output_dir=self.args.output_dir)

        # Acoustic args
        self._setup_acoustic_args()

        # Trainer states
        self.state = TrainerState(greater_is_better=self.args.greater_is_better)
        self.accelerator.register_for_checkpointing(self.state)  # Register accelerate objects

        # Pandas settings for better display
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        if self.accelerator.is_local_main_process:
            prepare_empty_dir(
                [self.checkpoints_dir, self.tb_log_dir, self.enhanced_dir, self.metrics_dir],
                resume=self.args.resume_from_checkpoint,
            )

            logger.info(f"\nEnvironment information:\n{print_env()}")

            self.writer = TensorboardLogger(self.tb_log_dir.as_posix())
            self.writer.log_config(self.args.to_dict())

            logger.info(f"Trainer file is saved to {self.trainer_args_path.as_posix()}.")
            self.args.save(self.trainer_args_path.as_posix())

            logger.info(f"Model arguments are saved to {self.model_args_path.as_posix()}.")
            if hasattr(self.model, "args"):
                self.model.args.save(self.model_args_path.as_posix())

            # Backup of project code
            # shutil.copytree(src=self.source_code_dir.as_posix(), dst=self.source_code_backup_dir.as_posix())
            # logger.info(f"Project code is backed up to {self.source_code_backup_dir.as_posix()}.")

            # Model summary
            logger.info(f"\n{summary(self.model, verbose=0)}")

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator = Accelerator(
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=self.args.ddp_find_unused_parameters)
            ],
        )

        self.rank = accelerator.device
        self.device = accelerator.device  # alias of rank
        self.num_devices = accelerator.num_processes
        self.accelerator = accelerator

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArgs) -> Tuple[torch.optim.Optimizer, Dict]:
        """Returns the optimizer class and optimizer parameters based on the training arguments."""
        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {"betas": (args.adam_beta1, args.adam_beta2), "eps": args.adam_epsilon}

        if args.optim == "adamw":
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs |= adam_kwargs
        elif args.optim == "adam":
            optimizer_cls = torch.optim.Adam
            optimizer_kwargs |= adam_kwargs
        else:
            raise ValueError(f"Unknown optimizer: {args.optim}")

        return optimizer_cls, optimizer_kwargs

    def create_optimizer(self):
        """Setup the optimizer."""
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

        return self.optimizer

    @staticmethod
    def get_warmup_steps(warmup_steps, max_steps, warmup_ratio):
        if warmup_steps > 0:
            logger.info(f"warmup_steps={warmup_steps}. warmup_ratio will be ignored.")
            return warmup_steps
        else:
            return math.ceil(max_steps * warmup_ratio)

    def create_warmup_scheduler(self, optimizer, scheduler_name, max_steps: int):
        num_warmup_steps = self.get_warmup_steps(self.args.warmup_steps, max_steps, self.args.warmup_ratio)
        if scheduler_name == "constant_schedule_with_warmup":
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif scheduler_name == "linear_schedule_with_warmup":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_steps
            )
        else:
            raise ValueError(f"Invalid scheduler name: {scheduler_name}")

    def create_scheduler(self, max_steps: int):
        """Create schedulers.

        You can override this method to create your own schedulers. For example, in GAN training, you may want to
        create two schedulers for the generator and the discriminator.

        Args:
            max_steps: the maximum number of steps to train.
        """
        self.lr_scheduler = self.create_warmup_scheduler(
            optimizer=self.optimizer, scheduler_name=self.args.lr_scheduler_type, max_steps=max_steps
        )
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup the optimizer and the learning rate scheduler."""
        self.create_optimizer()
        self.create_scheduler(num_training_steps)

    def _run_early_stop_check(self, score: float):
        should_stop = False

        if self._check_improvement(score, save_max_score=self.args.greater_is_better):
            self.state.best_score = score
            self.state.best_score_epoch = self.state.epochs_trained
            self._save_checkpoint(self.state.epochs_trained, is_best_epoch=True)
            self.state.early_stopping_patience_counter = 0
            logger.info(f"Found new best score: {score:.4f}, saving checkpoint...")
        else:
            logger.info(
                f"Score did not improve from {self.state.best_score:.4f} at epoch {self.state.best_score_epoch}."
            )
            self.state.early_stopping_patience_counter += 1
            logger.info(
                f"Early stopping counter: {self.state.early_stopping_patience_counter} out of {self.args.early_stopping_patience}"
            )

            if self.state.early_stopping_patience_counter >= self.args.early_stopping_patience:
                logger.info("Early stopping triggered, stopping training...")
                should_stop = True

        return should_stop

    def _setup_acoustic_args(self):
        """Setup acoustic arguments."""
        n_fft = self.args.acoustic_n_fft
        hop_length = self.args.acoustic_hop_length
        win_length = self.args.acoustic_win_length
        sr = self.args.acoustic_sr

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

    def _setup_exp_paths(self, output_dir):
        """Set the paths for the experiment.

        Args:
            save_dir: the root directory to save all experiments.

        Notes:
            - save_dir: /home/xhao/exp
            - checkpoints_dir: /home/xhao/exp/fullsubnet_lr_0.1/checkpoints
            - tb_log_dir: /home/xhao/exp/fullsubnet_lr_0.1/tb_log
            - src_source_code_dir: /home/xhao/audiozen
            - source_code_backup_dir: /home/xhao/exp/fullsubnet_lr_0.1/source_code__2023_01_07__17_19_57
            - config_path: /home/xhao/exp/fullsubnet_lr_0.1/config__2023_01_07__17_19_57.toml
        """
        time_now = self._get_time_now()

        self.output_dir = Path(output_dir).expanduser().absolute()
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.tb_log_dir = self.output_dir / "tb_log"
        self.enhanced_dir = self.output_dir / "enhanced"
        self.metrics_dir = self.output_dir / "metrics"

        # Each run will have a unique source code, config, and log file.
        self.source_code_dir = Path(__file__).expanduser().absolute().parent.parent.parent
        self.source_code_backup_dir = self.output_dir / f"source_code__{time_now}"
        self.trainer_args_path = self.output_dir / f"trainer_args__{time_now}.yaml"
        self.model_args_path = self.output_dir / f"model_args__{time_now}.yaml"

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

    def _load_checkpoint(self, ckpt):
        """load a checkpoint from the checkpints directory.

        Args:
            ckpt_path: "best", "latest", or a path to a checkpoint file
        """
        if ckpt == "best":
            ckpt = self.checkpoints_dir / "best"
        elif ckpt == "latest":
            ckpt = self._find_latest_ckpt_path()
        else:
            ckpt = Path(ckpt).expanduser().absolute()

        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt.as_posix()} not found.")

        self.accelerator.load_state(ckpt, map_location="cpu")

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

        # Find all regular checkpoints and only keep the latest `args.save_total_limit` regular checkpoints
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_*"))

        if epoch < len(checkpoints):
            logger.warning(
                f"Current epoch is {epoch}, but found {len(checkpoints)} checkpoints. "
                f"This may be caused by you running the same experiment multiple times. "
                f"Recommend to run the experiment with a different `output_dir`."
            )

        if len(checkpoints) > self.args.save_total_limit:
            logger.info(
                f"Found {len(checkpoints)} checkpoints, only keeping the latest {self.args.save_total_limit} checkpoints."
            )
            for checkpoint_dir in checkpoints[: -self.args.save_total_limit]:
                shutil.rmtree(checkpoint_dir.as_posix())
                logger.info(f"Checkpoint {checkpoint_dir.as_posix()} is removed.")

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

    def create_bar_desc(self, loss_dict: Dict[str, float]):
        bar_desc = ""
        for k, v in loss_dict.items():
            bar_desc += f"{k}: {(v):.4f}, "
        bar_desc += f"lr: {self.lr_scheduler.get_last_lr()[-1]:.10f}"

        if self.args.plot_lr:
            self.writer.add_scalar("Train_Step/lr", self.lr_scheduler.get_last_lr()[-1], self.state.steps_trained)

        return bar_desc

    def get_train_dataloader(self) -> DataLoader:
        """Create the training dataloader.

        Returns:
            train_dataloader: the training dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "worker_init_fn": seed_worker,
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloaders(self) -> Dict[str, DataLoader]:
        """Create the evaluation dataloaders.

        If the eval_dataset is a single dataset, it will be converted to a dictionary with the key "default".

        Returns:
            eval_dataloaders: the evaluation dataloaders.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a eval_dataset.")

        eval_dataset = self.eval_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": True,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if not isinstance(eval_dataset, dict):
            eval_dataset = {"default": eval_dataset}

        eval_dataloaders = {}
        for key, dataset in eval_dataset.items():
            eval_dataloaders[key] = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        return eval_dataloaders

    def train(self):
        """Main training entry point.

        Args:
            train_dataloader: the dataloader to train.
            evaluation_dataloaders: the dataloader(s) to validate.

        Notes:
            You are responsible for calling ``.backward()``, ``.step()``, ``clip_grad_norm_()``, and ``.zero_grad()``
            in your implementation of `training_step()`. Accelerate will automatically handle the gradient accumulation
            for you. It means that in gradient accumulation, the step() of optimizer and scheduler is called only when
            gradient_accumulation_steps is reached.

            The training step is implemented as follows:

            .. code-block:: python

                    self.optimizer.zero_grad()
                    loss = training_step(batch, batch_idx)
                    self.accelerator.backward(loss)
                    self.optimizer.step()

                    return {"loss": loss}
        """
        self.is_in_train = True
        if self.args.debug:
            logger.info("Debug mode is on")
            DebugUnderflowOverflow(self.model)

        self.accelerator.free_memory()

        # Create dataloaders
        train_dataloader = self.get_train_dataloader()

        # Training control variables
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = self.args.num_train_epochs

        # Create optimizer and scheduler
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        # Load potential model checkpoint
        resume_from_checkpoint = self.args.resume_from_checkpoint
        # Resume from the latest checkpoint
        if isinstance(resume_from_checkpoint, bool):
            if resume_from_checkpoint:
                self._load_checkpoint(ckpt="latest")
            else:
                logger.info("Training from scratch.")
        # Fine-tune from a specific checkpoint
        elif isinstance(resume_from_checkpoint, str):
            self._load_checkpoint(ckpt=resume_from_checkpoint)
        else:
            raise ValueError(f"Invalid ``resume_from_checkpoint``: {resume_from_checkpoint}")

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  `steps_per_epoch` = {num_update_steps_per_epoch:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")

        for epoch in range(self.state.epochs_trained + 1, num_train_epochs + 1):
            logger.info(f"{'=' * 9} Epoch {epoch} out of {num_train_epochs} {'=' * 9}")
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
                position=0,
                leave=True,
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
                    if self.accelerator.sync_gradients and self.accelerator.is_local_main_process:
                        bar_desc = self.create_bar_desc(loss_dict)
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
            if self.accelerator.is_local_main_process and epoch % self.args.save_epoch_interval == 0:
                self._save_checkpoint(epoch, is_best_epoch=False)

            if epoch % self.args.eval_epoch_interval == 0:
                with torch.no_grad():
                    logger.info("Training finished, begin evaluation...")
                    score = self.evaluate()

                    if self.accelerator.is_local_main_process:
                        should_stop = self._run_early_stop_check(score)
                        if should_stop:
                            self.accelerator.set_trigger()

                    logger.info("evaluation finished.")

            self.accelerator.wait_for_everyone()
            # If any process triggers early stopping, stop training
            if self.accelerator.check_trigger():
                break

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation and returns metrics.

        Returns:
            score: the metric score of the evaluation epoch.
        """
        logger.info("Begin evaluation...")
        eval_dataloaders = self.get_eval_dataloaders()

        if not self.is_in_train:
            self.model = self.accelerator.prepare(self.model)

            resume_from_checkpoint = self.args.resume_from_checkpoint
            # Resume from the latest checkpoint
            if isinstance(resume_from_checkpoint, bool):
                if resume_from_checkpoint:
                    self._load_checkpoint(ckpt="latest")
                else:
                    logger.info("Evaluating on a randomly initialized model.")
            # Resume from a specific checkpoint
            elif isinstance(resume_from_checkpoint, str):
                self._load_checkpoint(ckpt=resume_from_checkpoint)
            else:
                raise ValueError(f"Invalid ``resume_from_checkpoint``: {resume_from_checkpoint}")

        evaluation_output = self.evaluation_loop(eval_dataloaders, description="evaluate", gather_step_output=True)

        logger.info("Evaluation finished, begin hook `evaluate_epoch_end`...")
        if self.accelerator.is_local_main_process:
            # only the main process will run evaluation_epoch_end
            score = self.evaluation_epoch_end(evaluation_output)
            return score
        else:
            return None

    @torch.no_grad()
    def predict(self):
        logger.info("Begin predicting...")
        eval_dataloaders = self.get_eval_dataloaders()

        resume_from_checkpoint = self.args.resume_from_checkpoint
        # Resume from the latest checkpoint
        if isinstance(resume_from_checkpoint, bool):
            if resume_from_checkpoint:
                self._load_checkpoint(ckpt="latest")
            else:
                logger.info("Evaluating on a randomly initialized model.")
        # Resume from a specific checkpoint
        elif isinstance(resume_from_checkpoint, str):
            self._load_checkpoint(ckpt=resume_from_checkpoint)
        else:
            raise ValueError(f"Invalid ``resume_from_checkpoint``: {resume_from_checkpoint}")

        self.evaluation_loop(eval_dataloaders, description="predict", gather_step_output=False)

    @torch.no_grad()
    def evaluation_loop(self, dataloaders: Dict[str, DataLoader], description: str, gather_step_output: bool = False):
        """Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`."""
        args = self.args

        self.set_models_to_eval_mode()

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Batch size = {args.eval_batch_size}")

        evaluation_output = []
        for key, dataloader in dataloaders.items():
            dataloader_output = []
            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Evaluation on dataloader {key}",
                    bar_format="{l_bar}{r_bar}",
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_local_main_process,
                )
            ):
                # We recommend you directly calculate the metric score in the evaluation_step function and return the
                # metric score in the evaluation_step function, and then calculate the mean of the metric score
                # in the evaluation_epoch_end function.
                step_output = self.evaluation_step(batch, batch_idx, key)

                # If gather_step_output is True, we will gather the step_output from all processes and return a list of
                # step_output. If gather_step_output is False, we will return a list of gathered step_output.
                # If do predict, we don't need to gather the step_output from all processes.
                # If do evaluation, we need to do it.
                if gather_step_output:
                    """
                    [{
                        "metric_1": metric_1_score,
                        "metric_2": metric_1_score,
                        ...
                    }, ...]
                    """
                    step_output = self.accelerator.gather_for_metrics(step_output)
                dataloader_output.append(step_output)
            evaluation_output.append(dataloader_output)

        return evaluation_output

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
        """Implement the logic of the end of a training epoch. Please override this function if you want to do something.

        When the training epoch ends, this function will be called. The input is a list of the loss dict of each step
        in a training epoch. You may want to log the epoch-level training loss here.

        .. code-block:: python
            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    loss = training_step(batch, batch_idx)
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
        loss_keys = training_epoch_output[0].keys()

        # Compute mean loss on all loss items on a epoch
        for key in loss_keys:
            loss_items = [step_out[key] for step_out in training_epoch_output]
            loss_mean = torch.mean(torch.tensor(loss_items))

            if self.accelerator.is_local_main_process:
                logger.info(f"Loss '{key}' on epoch {self.state.epochs_trained}: {loss_mean}")
                self.writer.add_scalar(f"Train_Epoch/{key}", loss_mean, self.state.epochs_trained)

    def evaluation_step(self, batch, batch_idx, dataloader_idx):
        """Implement a evaluation/prediction step.

        This function defines the evaluation step. The input batch is from a eval dataloader.
        Here is the persuade code for validating a model:

        .. code-block:: python
            :emphasize-lines: 4

            evaluation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = evaluation_step(batch, batch_idx)
                    evaluation_epoch_output.append(loss_or_data)

            score = evaluation_epoch_end(evaluation_epoch_output)
            return score

        Notes:
            **The evaluation step will be run on all processes.**

            About batch size:
            If your evaluation data have the same length, you may use a batch size larger than 1 to speed up the evaluation.
            For example, if you have 1000 samples in the evaluation set, and you have a batch size of 100, then you will
            have 10 batches in the evaluation set. However, if your data in the evaluation set has a different length, please
            use a batch size of 1. It still works for distributed evaluation. Otherwise, you will get an error.

            About distributed evaluation:
            The output of this function will be gathered across all processes. For example, if you have 4 processes, and
            you have a batch size of 1, then you will have 4 outputs from this function. The output of this function will
            be gathered across all processes. The first dimension of the result is num_processes multiplied by the first
            dimension of the input tensors. **Please make sure the first dimension of the input tensors is the batch size.**
            **The last dimension of the output will be padded to the length of the longest sample in the evaluation set.**
            It means that the output will be a tensor with the shape of [num_processes * batch_size, max_length]. If you
            calculate the metric score on the output, you should do a truncation to remove the padding. Otherwise, if you
            are using a metric that sensitive to the padding, you will get a wrong metric score. It is not easy to
            implement this truncation in the ``evaluation_epoch_end`` function. We recommend you directly calculate the metric
            score in the evaluation_step function. I guess the Accelerate team will implement a automatic truncation in the
            future. https://github.com/huggingface/accelerate/issues/226

        Args:
            batch: a batch of data.
            batch_idx: the index of the batch.
            dataloader_idx: the index of the dataloader.

        Returns:
            output: the output of the batch. It may enhanced audio signals.
        """
        raise NotImplementedError

    def evaluation_epoch_end(self, evaluation_epoch_output):
        """evaluation epoch end.

        The input `evaluation_epoch_output` will be a list of list. For example, if you have two dataloaders, the `evaluation_epoch_output` will be:

        .. code-block:: python

            evaluation_epoch_output = [
                [dataloader_1_batch_1_output, dataloader_1_batch_2_output, ...],
                [dataloader_2_batch_1_output, dataloader_2_batch_2_output, ...],
                ...,
            ]


        The output of this function should be a metric score, which will be used to determine whether the current model is the best model.

        .. code-block:: python
            :emphasize-lines: 7

            evaluation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = evaluation_step(batch, batch_idx)
                    evaluation_epoch_output.append(loss_or_data)

            score = evaluation_epoch_end(evaluation_epoch_output)
            return score

        Args:
            evaluation_epoch_output: the output of the evaluation epoch. It is a list of list.

        Returns:
            score: the metric score of the evaluation epoch.
        """
        raise NotImplementedError
