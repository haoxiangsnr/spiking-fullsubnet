import logging

import torch

from audiozen.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class EarlyStopTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.patience = 10
        self.should_stop = False
        self.wait_count = 0
        self.mode = "min"

    def _evaluate_stopping_criteria(self):
        pass

    def run_early_stopping_check(self, current_score):
        """Check if the current score is better than the best score.

        This function is runned on the end of each epoch.
        """
        if current_score < self.best_score:
            self.wait_count += 1
            self.logger.info(
                f"Early stopping counter: {self.wait_count} out of {self.patience}"
            )

            if self.wait_count >= self.patience:
                self.logger.info("Early stopping triggered")
                self.should_stop = True

    def _create_state_dict(self, epoch):
        """Create a state dict for saving."""
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict(),  # fmt: skip
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "wait_count": self.wait_count,
        }

        return state_dict

    def _load_state_dict(self, checkpoint_dict):
        self.start_epoch = checkpoint_dict["epoch"] + 1
        self.best_score = checkpoint_dict["best_score"]
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.scaler.load_state_dict(checkpoint_dict["scaler"])
        self.wait_count = checkpoint_dict["wait_count"]
